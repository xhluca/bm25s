import unittest

import numpy as np

import bm25s
from bm25s.tokenization import Tokenizer


class TestBM25SLoadingSaving(unittest.TestCase):
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
