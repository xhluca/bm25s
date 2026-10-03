import unittest

import numpy as np

import bm25s
from bm25s import selection as selection_np


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


if __name__ == "__main__":
    unittest.main()
