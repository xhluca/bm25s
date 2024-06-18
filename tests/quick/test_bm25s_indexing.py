import json
import os
import time
import unittest
from pathlib import Path
import warnings
import logging

import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
import Stemmer

import bm25s

def check_scores_all_close(score1, score2, **kwargs):
    for key in score1.keys():
        matrix1 = score1[key]
        matrix2 = score2[key]
            
        if matrix1.shape != matrix2.shape:
            return False
        if not np.allclose(matrix1, matrix2, **kwargs):
            return False

        return True

class BM25SIndexing(unittest.TestCase):
    def test_indexing_by_corpus_type(self):
        warnings.filterwarnings("ignore", category=ResourceWarning)
        class Tokenized:
            def __init__(self, ids, vocab):
                self.ids = ids
                self.vocab = vocab

        dataset = "scifact"
        rel_save_dir = "datasets"
        # Download and prepare dataset
        base_url = (
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
        )
        url = base_url.format(dataset)
        out_dir = Path(__file__).parent / rel_save_dir
        data_path = download_and_unzip(url, str(out_dir))

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        corpus_ids, corpus_lst = [], []
        for key, val in corpus.items():
            corpus_ids.append(key)
            corpus_lst.append(val["title"] + " " + val["text"])

        stemmer = Stemmer.Stemmer("english")
        corpus_tokens_lst = bm25s.tokenize(
            corpus_lst,
            stopwords="en",
            stemmer=stemmer,
            leave=False,
            return_ids=False,
        )

        corpus_tokenized = bm25s.tokenize(
            corpus_lst,
            stopwords="en",
            stemmer=stemmer,
            leave=False,
            return_ids=True,
        )

        bm25_tokens = bm25s.BM25(k1=0.9, b=0.4)
        bm25_tokens.index(corpus_tokens_lst)

        bm25_tuples = bm25s.BM25(k1=0.9, b=0.4)
        bm25_tuples.index((corpus_tokenized.ids, corpus_tokenized.vocab))

        bm25_objects = bm25s.BM25(k1=0.9, b=0.4)
        bm25_objects.index(
            Tokenized(ids=corpus_tokenized.ids, vocab=corpus_tokenized.vocab)
        )

        bm25_namedtuple = bm25s.BM25(k1=0.9, b=0.4)
        named_tuple = bm25s.tokenization.Tokenized(
            ids=corpus_tokenized.ids, vocab=corpus_tokenized.vocab
        )
        bm25_namedtuple.index(named_tuple)

        # now, verify that the sparse matrix matches
        self.assertTrue(
            check_scores_all_close(
                bm25_tokens.scores, bm25_tuples.scores
            ),
            "Tokenized and Tuple indexing do not match",
        )
        self.assertTrue(
            check_scores_all_close(
                bm25_tokens.scores, bm25_objects.scores
            ),
            "Tokenized and Object indexing do not match",
        )
        self.assertTrue(
            check_scores_all_close(
                bm25_tokens.scores, bm25_namedtuple.scores
            ),
            "Tokenized and NamedTuple indexing do not match",
        )
