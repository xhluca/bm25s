import unittest

import numpy as np

import bm25s


class TestNumbaQuantize(unittest.TestCase):
    """quantize=True retrieves with 8-bit impacts: scores are approximate
    (within the quantization tolerance) and the returned documents must
    closely match the exact retrieval."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        n_vocab = 500
        corpus = [(rng.zipf(1.4, size=20) % n_vocab).tolist() for _ in range(20000)]
        cls.model = bm25s.BM25(backend="numba")
        cls.model.index((corpus, {i: i for i in range(n_vocab)}), show_progress=False)
        cls.queries = [(rng.zipf(1.3, size=6) % n_vocab).tolist() for _ in range(40)]

    def test_quantized_close_to_exact(self):
        k = 50
        exact_docs, exact_scores = self.model.retrieve(
            self.queries, k=k, show_progress=False
        )
        q_docs, q_scores = self.model.retrieve(
            self.queries, k=k, show_progress=False, quantize=True
        )

        step = float(self.model.scores["q_step"])
        n_terms = max(len(q) for q in self.queries)
        # every document score is off by at most step/2 per query term
        tol = step * n_terms

        for i in range(len(self.queries)):
            # the document sets may legitimately differ among (near-)ties, but
            # both runs must retrieve documents of equivalent score quality:
            # the quantized cutoff cannot fall below the exact cutoff by more
            # than the quantization tolerance
            self.assertGreaterEqual(q_scores[i, -1], exact_scores[i, -1] - 2 * tol)

            # rank-by-rank, de-quantized scores stay within tolerance
            self.assertTrue(np.all(np.abs(q_scores[i] - exact_scores[i]) <= 2 * tol))

            exact_set = set(exact_docs[i].tolist())
            quant_set = set(q_docs[i].tolist())
            self.assertGreaterEqual(len(exact_set & quant_set) / k, 0.5)

        # scores are sorted
        self.assertTrue(np.all(np.diff(q_scores, axis=1) <= 0))

    def test_quantized_data_cached(self):
        self.model.retrieve(self.queries[:2], k=5, show_progress=False, quantize=True)
        self.assertIn("data_q", self.model.scores)
        self.assertEqual(self.model.scores["data_q"].dtype, np.uint8)

    def test_model_level_flag(self):
        model = bm25s.BM25(backend="numba", quantize=True)
        model.index(
            ([[1, 2, 3], [2, 3, 4], [3, 4, 5]] * 100, {i: i for i in range(6)}),
            show_progress=False,
        )
        docs, scores = model.retrieve([[2, 3]], k=3, show_progress=False)
        self.assertIn("data_q", model.scores)
        self.assertEqual(docs.shape, (1, 3))

    def test_unsupported_combinations(self):
        with self.assertRaises(ValueError):
            self.model.retrieve(
                self.queries[:2], k=5, show_progress=False,
                quantize=True, exact_ties=True,
            )
        with self.assertRaises(ValueError):
            mask = np.ones(self.model.scores["num_docs"], dtype=np.float32)
            self.model.retrieve(
                self.queries[:2], k=5, show_progress=False,
                quantize=True, weight_mask=mask,
            )
        numpy_model = bm25s.BM25(backend="numpy")
        numpy_model.index(
            ([[1, 2], [2, 3]], {i: i for i in range(4)}), show_progress=False
        )
        with self.assertRaises(ValueError):
            numpy_model.retrieve([[1]], k=1, show_progress=False, quantize=True)


if __name__ == "__main__":
    unittest.main()
