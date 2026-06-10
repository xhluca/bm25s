import unittest

import numpy as np

import bm25s


class TestNumbaExactTies(unittest.TestCase):
    """The numba backend selects tied scores arbitrarily by default; with
    exact_ties=True it must reproduce the exhaustive scan bit for bit."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        n_vocab = 400
        # every document repeated many times so that tied scores are pervasive
        base = [(rng.zipf(1.4, size=12) % n_vocab).tolist() for _ in range(300)]
        corpus = [list(base[i % 300]) for i in range(30000)]
        cls.model = bm25s.BM25(backend="numba")
        cls.model.index((corpus, {i: i for i in range(n_vocab)}), show_progress=False)
        cls.queries = [(rng.zipf(1.3, size=6) % n_vocab).tolist() for _ in range(40)]
        cls.num_docs = cls.model.scores["num_docs"]

    def _exhaustive_reference(self, k, weight_mask=None):
        from bm25s.numba.retrieve_utils import (
            _retrieve_internal_jitted_parallel_nonoccurrence,
        )

        qp = np.cumsum([0] + [len(q) for q in self.queries], dtype=np.int32)
        qf = np.concatenate(self.queries).astype(np.int32)
        sc = self.model.scores
        return _retrieve_internal_jitted_parallel_nonoccurrence(
            query_tokens_ids_flat=qf,
            query_pointers=qp,
            k=k,
            sorted=True,
            dtype=np.dtype("float32"),
            int_dtype=np.dtype("int32"),
            data=sc["data"],
            indptr=sc["indptr"],
            indices=sc["indices"],
            num_docs=sc["num_docs"],
            nonoccurrence_array=None,
            weight_mask=weight_mask,
        )

    def test_exact_ties_bit_identical(self):
        for k in [1, 10, 1000]:
            ref_scores, ref_inds = self._exhaustive_reference(k)
            res = self.model.retrieve(
                self.queries, k=k, show_progress=False, exact_ties=True
            )
            np.testing.assert_array_equal(res.scores, ref_scores)
            np.testing.assert_array_equal(res.documents, ref_inds)

    def test_default_mode_same_scores(self):
        for k in [1, 10, 1000]:
            ref_scores, _ = self._exhaustive_reference(k)
            res = self.model.retrieve(self.queries, k=k, show_progress=False)
            np.testing.assert_array_equal(
                np.sort(res.scores, axis=1), np.sort(ref_scores, axis=1)
            )
            # retrieved documents must be unique per query
            for row in res.documents:
                self.assertEqual(len(set(row.tolist())), len(row))

    def test_exact_ties_with_weight_mask(self):
        rng = np.random.default_rng(0)
        mask = (rng.random(self.num_docs) > 0.5).astype(np.float32)
        ref_scores, ref_inds = self._exhaustive_reference(10, weight_mask=mask)
        res = self.model.retrieve(
            self.queries, k=10, show_progress=False, weight_mask=mask, exact_ties=True
        )
        np.testing.assert_array_equal(res.scores, ref_scores)
        np.testing.assert_array_equal(res.documents, ref_inds)


if __name__ == "__main__":
    unittest.main()
