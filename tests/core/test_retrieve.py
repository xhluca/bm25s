import os
import shutil
from pathlib import Path
import unittest
import tempfile

import numpy as np
import bm25s
import Stemmer  # optional: for stemming

class TestBM25SLoadingSaving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create your corpus here
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]

        # optional: create a stemmer
        stemmer = Stemmer.Stemmer("english")

        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25(method='bm25+')
        retriever.index(corpus_tokens)

        # Save the retriever to temp dir
        cls.retriever = retriever
        cls.corpus = corpus
        cls.corpus_tokens = corpus_tokens
        cls.stemmer = stemmer
    
    def test_retrieve(self):
        ground_truth = np.array([[0, 2]])

        # first, try with default mode
        query = "a cat is a feline, it's sometimes beautiful but cannot fly"
        query_tokens_obj = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=True)

        # retrieve the top 2 documents
        results = self.retriever.retrieve(query_tokens_obj, k=2).documents
        
        # assert that the retrieved indices are correct
        self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

        # now, try tokenizing with text tokens
        query_tokens_texts = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False)
        results = self.retriever.retrieve(query_tokens_texts, k=2).documents
        self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

        # now, try to pass a tuple of tokens
        ids, vocab = query_tokens_obj
        query_tokens_tuple = (ids, vocab)
        results = self.retriever.retrieve(query_tokens_tuple, k=2).documents
        self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

        # finally, try to pass a 2-tuple of tokens with text tokens to "try to trick the system"
        queries_as_tuple = (query_tokens_texts[0], query_tokens_texts[0])
        # only retrieve 1 document
        ground_truth = np.array([[0], [0]])
        results = self.retriever.retrieve(queries_as_tuple, k=1).documents
        self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

    def test_failure_of_bad_tuple(self):
        # try to pass a tuple of tokens with different lengths
        query = "a cat is a feline, it's sometimes beautiful but cannot fly"
        query_tokens_obj = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=True)
        query_tokens_texts = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False)
        ids, vocab = query_tokens_obj
        query_tokens_tuple = (vocab, ids)

        with self.assertRaises(ValueError):
            self.retriever.retrieve(query_tokens_tuple, k=2)
        
        # now, test if there's vocab twice or ids twice
        query_tokens_tuple = (ids, ids)
        with self.assertRaises(ValueError):
            self.retriever.retrieve(query_tokens_tuple, k=2)

        # finally, test only passing vocab
        query_tokens_tuple = (vocab, )
        with self.assertRaises(ValueError):
            self.retriever.retrieve(query_tokens_tuple, k=2)

        



    @classmethod
    def tearDownClass(cls):
        pass