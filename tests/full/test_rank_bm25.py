import unittest

from .. import BM25TestCase

class TestRankBM25Full(BM25TestCase):
    def test_bm25_sparse_vs_rank_bm25_on_nfcorpus(self):
        self.compare_with_rank_bm25("nfcorpus")
    
    def test_bm25_sparse_vs_rank_bm25_on_scifact(self):
        self.compare_with_rank_bm25("scifact")
    
    def test_bm25_sparse_vs_rank_bm25_on_scidocs(self):
        self.compare_with_rank_bm25("scidocs")

    # fiqa
    def test_bm25_sparse_vs_rank_bm25_on_fiqa(self):
        self.compare_with_rank_bm25("fiqa")

    # arguana
    def test_bm25_sparse_vs_rank_bm25_on_arguana(self):
        self.compare_with_rank_bm25("arguana")

if __name__ == '__main__':
    unittest.main()
