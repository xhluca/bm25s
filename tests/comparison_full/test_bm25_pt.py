import unittest

from .. import BM25TestCase

class TestBM25PTQuick(BM25TestCase):
    def test_bm25_sparse_vs_rank_bm25_on_nfcorpus(self):
        self.compare_with_bm25_pt("nfcorpus")
    
    def test_bm25_sparse_vs_rank_bm25_on_scifact(self):
        self.compare_with_bm25_pt("scifact")
    
    def test_bm25_sparse_vs_rank_bm25_on_scidocs(self):
        self.compare_with_bm25_pt("scidocs")

    # fiqa
    def test_bm25_sparse_vs_rank_bm25_on_fiqa(self):
        self.compare_with_bm25_pt("fiqa")

    # arguana
    def test_bm25_sparse_vs_rank_bm25_on_arguana(self):
        self.compare_with_bm25_pt("arguana")

if __name__ == '__main__':
    unittest.main()
