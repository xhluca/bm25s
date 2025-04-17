import os
import shutil
from pathlib import Path
import unittest
import numpy as np
import tempfile
import Stemmer  # optional: for stemming
import unittest.mock
import json

import bm25s

class TestVocabDict(unittest.TestCase):
    def test_vocab_dict(self):

        # Create the BM25 model and index the corpus
        stemmer = Stemmer.Stemmer("english")
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]
        # Note: allow_empty=False will ensure that "" is not in the vocab_dict
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, allow_empty=False)

        # check that "" is not in the vocab
        self.assertFalse("" in corpus_tokens.vocab)

        # 1. index(corpus,create_empty_token=True) --> correct
        retriever = bm25s.BM25(method='bm25+')
        retriever.index(corpus_tokens, create_empty_token=True)
        self.assertTrue(retriever.vocab_dict is not None)
        self.assertTrue("" in retriever.vocab_dict)

        self.assertEqual(len(retriever.vocab_dict), len(retriever.unique_token_ids_set))
        self.assertEqual(set(retriever.vocab_dict.values()), set(retriever.unique_token_ids_set))

        # empty_sentence = ["", "", ""]
        # empty_sentence_tokens = bm25s.tokenize(empty_sentence, stopwords="en", stemmer=stemmer, allow_empty=True)
        # check that "" is in the vocab_dict
        
        # 2. index(corpus,create_empty_token=True) --> throwing an error
        # pass

        # 3. index(corpus,create_empty_token=False) --> correct
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, allow_empty=False)

        retriever = bm25s.BM25(method='lucene')
        retriever.index(corpus_tokens, create_empty_token=False)
        self.assertTrue(retriever.vocab_dict is not None)
        self.assertFalse("" in retriever.vocab_dict)

        self.assertEqual(len(retriever.vocab_dict), len(retriever.unique_token_ids_set))
        self.assertEqual(set(retriever.vocab_dict.values()), set(retriever.unique_token_ids_set))

        # 4. index(corpus,create_empty_token=False) --> throwing an error
        retriever = bm25s.BM25(method='bm25+')
        retriever.index(corpus_tokens, create_empty_token=False)

        new_docs = ["cat", "", "potato"]

        tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=stemmer)
        corpus_tokens = tokenizer.tokenize(corpus, return_as='ids', allow_empty=True)
        new_docs_tokens = tokenizer.tokenize(new_docs, return_as='ids', allow_empty=True)

        # create_empty_token=True
        retriever = bm25s.BM25(method='bm25+')
        retriever.index(corpus_tokens, create_empty_token=True)
        retriever.retrieve(new_docs_tokens, k=1)

        # create_empty_token=False
        retriever = bm25s.BM25(method='bm25+')
        # assert that this will throw an error
        with self.assertRaises(IndexError):
            retriever.index(corpus_tokens, create_empty_token=False)
            retriever.retrieve(new_docs_tokens, k=1)