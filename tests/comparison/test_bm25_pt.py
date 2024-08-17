import unittest

from .. import BM25TestCase

class TestBM25PTQuick(BM25TestCase):
    def test_bm25_sparse_vs_rank_bm25_on_nfcorpus(self):
        self.compare_with_bm25_pt("nfcorpus", corpus_subsample=4000, queries_subsample=1000)
    
    def test_bm25_sparse_vs_rank_bm25_on_scifact(self):
        self.compare_with_bm25_pt("scifact", corpus_subsample=4000, queries_subsample=1000)
    
    def test_bm25_sparse_vs_rank_bm25_on_scidocs(self):
        self.compare_with_bm25_pt("scidocs", corpus_subsample=4000, queries_subsample=1000)

    # fiqa
    def test_bm25_sparse_vs_rank_bm25_on_fiqa(self):
        self.compare_with_bm25_pt("fiqa", corpus_subsample=4000, queries_subsample=1000)

    # arguana
    def test_bm25_sparse_vs_rank_bm25_on_arguana(self):
        self.compare_with_bm25_pt("arguana", corpus_subsample=4000, queries_subsample=1000)

if __name__ == '__main__':
    unittest.main()
