from pathlib import Path
import warnings
import unittest

import numpy as np
import Stemmer
import beir.util
from beir.datasets.data_loader import GenericDataLoader

import bm25s
from bm25s.utils.beir import BASE_URL


class TestBM25PTQuick(unittest.TestCase):
    def test_bm25_sparse_vs_rank_bm25_on_nfcorpus(self):
        data_dir="datasets"
        index_dir = "bm25s_indices"
        dataset="scifact"
        split = "test"

        index_path = Path(index_dir) / dataset
        data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), data_dir)

        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Download and prepare dataset
        corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split=split)
        
        corpus_ids, corpus_lst = [], []
        for key, val in corpus.items():
            corpus_ids.append(key)
            corpus_lst.append(val["title"] + " " + val["text"])
        query_lst = list(queries.values())
        
        stemmer = Stemmer.Stemmer("english")
        model = bm25s.BM25(corpus=corpus)
        corpus_tokens = bm25s.tokenize(corpus_lst, stemmer=stemmer)
        model.index(corpus_tokens, show_progress=False)

        # Save the model
        model.save(index_path)

        # Load the model
        q_tokens = bm25s.tokenize(query_lst, stemmer=stemmer)
        model1 = bm25s.BM25.load(index_path, mmap=False, load_corpus=True)
        model2 = bm25s.BM25.load(index_path, mmap=True, load_corpus=True)
        
        res1 = model1.retrieve(q_tokens, show_progress=False)
        res2 = model2.retrieve(q_tokens, show_progress=False)

        # make sure the results are the same
        self.assertTrue(np.all(res1.scores == res2.scores))
        self.assertTrue(np.all(res1.documents == res2.documents))
