import unittest

import numpy as np

import bm25s
from bm25s import selection as selection_np
from bm25s.tokenization import Tokenized


class TestBM25SEmptyCorpus(unittest.TestCase):
    """Indexing an empty corpus (no documents, or only token-less documents).

    Previously this crashed in ``index`` with ``ValueError: max() iterable
    argument is empty`` and emitted ``RuntimeWarning: Mean of empty slice`` /
    ``invalid value encountered in scalar divide``. It should now build a valid
    empty index instead.
    """

    def assert_empty_index_well_formed(self, retriever: "bm25s.BM25", num_docs: int) -> None:
        self.assertEqual(retriever.scores["num_docs"], num_docs)
        # The empty token is seeded at id 0 so token-less queries have a target.
        self.assertEqual(retriever.vocab_dict.get(""), 0)
        # The scored matrix has zero populated entries.
        self.assertEqual(retriever.scores["data"].size, 0)

    def test_index_no_documents_does_not_crash(self) -> None:
        retriever = bm25s.BM25(backend="numpy")
        with np.errstate(all="raise"):  # no NaN/divide warnings allowed
            retriever.index([], show_progress=False)
        self.assert_empty_index_well_formed(retriever, num_docs=0)

    def test_index_only_token_less_documents_does_not_crash(self) -> None:
        retriever = bm25s.BM25(backend="numpy")
        with np.errstate(all="raise"):
            retriever.index([[], [], []], show_progress=False)
        self.assert_empty_index_well_formed(retriever, num_docs=3)

    def test_retrieve_from_empty_corpus_raises_clear_error(self) -> None:
        # k must not exceed the (zero) document count; the existing guard gives
        # a clear, actionable ValueError rather than an internal IndexError.
        retriever = bm25s.BM25(backend="numpy")
        retriever.index([], show_progress=False)
        with self.assertRaises(ValueError):
            retriever.retrieve(bm25s.tokenize("foo", show_progress=False), k=1, show_progress=False)

    def test_topk_of_empty_scores_returns_empty(self) -> None:
        # An empty score vector must yield aligned empty results, not
        # IndexError: cannot do a non-empty take from an empty axes.
        empty = np.empty(0, dtype=np.float32)
        scores, indices = selection_np.topk(empty, k=3, backend="numpy", sorted=True)
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(indices.shape, (0,))

    def test_token_less_corpus_serves_zero_scores(self) -> None:
        # Regression: a token-less corpus must reserve a real zero-score column
        # for the seeded empty token, so an ordinary (unmatched) query does not
        # trip the "max token ID higher than the number of tokens" guard and
        # instead returns every document with a zero score.
        for backend in ("numpy", "numba"):
            retriever = bm25s.BM25(backend="numpy")
            retriever.index([[], [], []], show_progress=False)
            # The empty token now owns column 0: indptr has one column (len 2).
            self.assertEqual(len(retriever.scores["indptr"]), 2)
            docs, scores = retriever.retrieve(
                [["hello"]], k=2, backend_selection=backend, show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            np.testing.assert_array_equal(scores, 0.0)
            # An empty-token query behaves the same way.
            docs, scores = retriever.retrieve(
                [[""]], k=2, backend_selection=backend, show_progress=False
            )
            np.testing.assert_array_equal(scores, 0.0)

    def test_token_less_corpus_survives_eager_and_mmap_roundtrip(self) -> None:
        import tempfile

        retriever = bm25s.BM25(backend="numpy")
        retriever.index([[], [], []], show_progress=False)
        save_dir = tempfile.mkdtemp()
        retriever.save(save_dir)
        for mmap in (False, True):
            loaded = bm25s.BM25.load(save_dir, mmap=mmap)
            docs, scores = loaded.retrieve(
                [["hello"]], k=2, backend_selection="numpy", show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            np.testing.assert_array_equal(scores, 0.0)

    def test_indexing_same_tokenized_twice_is_stable(self) -> None:
        # Indexing the same token-less Tokenized input twice must not mutate the
        # vocabulary or widen indptr between builds.
        tok = Tokenized(ids=[[], [], []], vocab={"": 0})
        retriever = bm25s.BM25(backend="numpy")
        retriever.index(tok, show_progress=False)
        indptr_first = retriever.scores["indptr"].copy()
        retriever.index(tok, show_progress=False)
        np.testing.assert_array_equal(retriever.scores["indptr"], indptr_first)
        self.assertEqual(retriever.vocab_dict, {"": 0})

    def test_token_less_corpus_nonoccurrence_is_consistent(self) -> None:
        # BM25L / BM25+ build a nonoccurrence array; it must line up with the
        # single reserved column rather than be empty.
        for method in ("bm25l", "bm25+"):
            retriever = bm25s.BM25(method=method, backend="numpy")
            retriever.index([[], [], []], show_progress=False)
            self.assertEqual(len(retriever.nonoccurrence_array), 1)
            docs, scores = retriever.retrieve(
                [["hello"]], k=1, backend_selection="numpy", show_progress=False
            )
            np.testing.assert_array_equal(scores, 0.0)


if __name__ == "__main__":
    unittest.main()
