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


    def test_retrieve_with_weight_mask(self):
        

        # first, try with default mode
        query = "cat feline dog bird fish"  # weights should be [2, 1, 1, 1], but after masking should be [2, 0, 0, 1]

        for dt in [np.float32, np.int32, np.bool_]:
            weight_mask = np.array([1, 0, 0, 1], dtype=dt)
            ground_truth = np.array([[0, 3]])

            query_tokens_obj = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=True)

            # retrieve the top 2 documents
            results = self.retriever.retrieve(query_tokens_obj, k=2, weight_mask=weight_mask).documents
            
            # assert that the retrieved indices are correct
            self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

            # now, try tokenizing with text tokens
            query_tokens_texts = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False)
            results = self.retriever.retrieve(query_tokens_texts, k=2, weight_mask=weight_mask).documents
            self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

            # now, try to pass a tuple of tokens
            ids, vocab = query_tokens_obj
            query_tokens_tuple = (ids, vocab)
            results = self.retriever.retrieve(query_tokens_tuple, k=2, weight_mask=weight_mask).documents
            self.assertTrue(np.array_equal(ground_truth, results), f"Expected {ground_truth}, got {results}")

            # finally, try to pass a 2-tuple of tokens with text tokens to "try to trick the system"
            queries_as_tuple = (query_tokens_texts[0], query_tokens_texts[0])
            # only retrieve 1 document
            ground_truth = np.array([[0], [0]])
            results = self.retriever.retrieve(queries_as_tuple, k=1, weight_mask=weight_mask).documents
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

    def test_value_error_for_very_small_corpus(self):
        query = "a cat is a feline, it's sometimes beautiful but cannot fly"
        query_tokens = bm25s.tokenize(
            [query], stopwords="en",
            stemmer=self.stemmer, return_ids=True
        )
        corpus_size = len(self.corpus)
        for k in range(0, 10):
            if k > corpus_size:
                with self.assertRaises(ValueError) as context:
                    self.retriever.retrieve(query_tokens, k=k)
                exception_str_should_include =\
                    "Please set with a smaller k or increase the size of corpus."
                self.assertIn(
                    exception_str_should_include,
                    str(context.exception),
                    f"[k={k}] Expected ValueError mentioning (but did not)"
                    f"; {exception_str_should_include}"
                )
            else:
                results, scores = self.retriever.retrieve(query_tokens, k=k)
                self.assertEqual(
                    int(results.size), k,
                    f"[k={k}] The number of searched items"
                    f" should be {k}; but it was {results.size}"
                )
                self.assertEqual(
                    int(scores.size), k,
                    f"[k={k}] The number of searched items"
                    f" should be {k}; but it was {scores.size}"
                )

    @classmethod
    def tearDownClass(cls):
        pass