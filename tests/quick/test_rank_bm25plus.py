import unittest

from .. import BM25TestCase


class TestRankBM25Quick(BM25TestCase):
    def test_bm25_sparse_vs_rank_bm25_on_nfcorpus(self):
        self.compare_with_rank_bm25(
            "nfcorpus", corpus_subsample=2000, queries_subsample=200, method="bm25+"
        )

    def test_bm25_sparse_vs_rank_bm25_on_scifact(self):
        self.compare_with_rank_bm25(
            "scifact", corpus_subsample=2000, queries_subsample=200, method="bm25+"
        )

    def test_bm25_sparse_vs_rank_bm25_on_scidocs(self):
        self.compare_with_rank_bm25(
            "scidocs", corpus_subsample=2000, queries_subsample=200, method="bm25+"
        )

    # fiqa
    def test_bm25_sparse_vs_rank_bm25_on_fiqa(self):
        self.compare_with_rank_bm25(
            "fiqa", corpus_subsample=2000, queries_subsample=200, method="bm25+"
        )

    # arguana
    def test_bm25_sparse_vs_rank_bm25_on_arguana(self):
        self.compare_with_rank_bm25(
            "arguana", queries_subsample=100, corpus_subsample=1000, method="bm25+"
        )


if __name__ == "__main__":
    unittest.main()
