"""
Miscellaneous tests for the tokenizer module.
"""

import unittest

import numpy as np
import Stemmer

import bm25s
from bm25s.tokenization import Tokenizer


class TestBM25SNewIds(unittest.TestCase):
    def test_empty_string(self):
        # Create an empty corpus
        corpus = ["", "", "", ""]
        # Create a list of queries
        queries = ["what is the meaning of life?"]

        # The tokenizer will return a list of list of tokens
        tokenizer = Tokenizer()
        corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple", allow_empty=True)

        self.assertEqual(
            corpus_tokens,
            bm25s.tokenization.Tokenized(ids=[[0], [0], [0], [0]], vocab={"": 0}),
            msg=f"Corpus tokens differ from expected: {corpus_tokens}",
        )

        query_tokens = tokenizer.tokenize(
            queries, return_as="ids", update_vocab=False, allow_empty=True
        )

        self.assertEqual(
            [[0]],
            query_tokens,
            msg=f"Query tokens differ from expected: {query_tokens}",
        )

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
        self.assertTrue(
            np.all(results == np.array([["", ""]])),
            msg=f"Results differ from expected: {results}, {scores}",
        )

    def test_new_ids(self):
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]

        tokenizer = Tokenizer(
            stemmer=None, stopwords=None, splitter=lambda x: x.split()
        )
        corpus_tokens = tokenizer.tokenize(corpus, allow_empty=False)

        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, create_empty_token=False)

        query = "What is a fly?"
        query_tokens = tokenizer.tokenize([query], update_vocab=True, allow_empty=False)
        self.assertListEqual([[27, 2, 0, 28]], query_tokens)

        results, scores = bm25.retrieve(query_tokens, k=3)
        self.assertTrue(
            np.all(np.array([[0, 2, 3]]) == results),
            msg=f"Results differ from expected: {results}, {scores}",
        )

    def test_failing_after_adding_new_tokens_query(self):
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]

        tokenizer = Tokenizer(
            stemmer=None, stopwords=None, splitter=lambda x: x.split()
        )
        corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple", allow_empty=False)

        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, create_empty_token=False)

        query = "unknownword"
        query_tokens = tokenizer.tokenize([query], update_vocab=True, allow_empty=False)

        # assert a valueError is raised
        with self.assertRaises(ValueError):
            results, scores = bm25.retrieve(query_tokens, k=3)
    
    def test_only_unknown_token_query(self):
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]

        tokenizer = Tokenizer(
            stemmer=None, stopwords=None, splitter=lambda x: x.split()
        )
        corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple")

        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens)

        query = "unknownword"
        query_tokens = tokenizer.tokenize([query], update_vocab=False)

        results, scores = bm25.retrieve(query_tokens, k=3)
        self.assertTrue(np.all(scores == 0.0))

    def test_only_unknown_token_query_stemmed(self):
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]

        stemmer = Stemmer.Stemmer("english")

        tokenizer = Tokenizer(
            stemmer=stemmer, stopwords=None, splitter=lambda x: x.split()
        )
        corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple")

        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens)

        query = "unknownword"
        query_tokens = tokenizer.tokenize([query], update_vocab=False)

        results, scores = bm25.retrieve(query_tokens, k=3)
        self.assertTrue(np.all(scores == 0.0))