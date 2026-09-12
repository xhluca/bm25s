import os
import shutil
from pathlib import Path
import unittest
import tempfile

import numpy as np
import bm25s
import Stemmer  # optional: for stemming


class TestRetrieveCorpusItems(unittest.TestCase):
    backend = "numpy"

    def setUp(self):
        self.retriever = bm25s.BM25(backend=self.backend, auto_compile=False)
        self.retriever.index([["cat"], ["dog"], ["fish"]], show_progress=False)
        self.queries = [["cat"], ["fish"]]
        self.expected = self.retriever.retrieve(self.queries, k=2, show_progress=False)
        self.assertTrue(np.issubdtype(self.expected.documents.dtype, np.integer))

    def assert_corpus_results(self, retriever, corpus):
        for return_as in ("tuple", "documents"):
            with self.subTest(return_as=return_as):
                result = retriever.retrieve(
                    self.queries,
                    corpus=corpus,
                    k=2,
                    return_as=return_as,
                    show_progress=False,
                )
                documents = result.documents if return_as == "tuple" else result
                self.assertEqual(documents.shape, self.expected.documents.shape)
                if isinstance(corpus, np.ndarray):
                    self.assertEqual(documents.dtype, corpus.dtype)
                else:
                    self.assertEqual(documents.dtype, np.dtype(object))
                for position in np.ndindex(documents.shape):
                    doc_id = self.expected.documents[position]
                    expected_document = corpus[int(doc_id)]
                    self.assertEqual(documents[position], expected_document)
                    self.assertIs(type(documents[position]), type(expected_document))
                    if isinstance(expected_document, (list, tuple, dict)) and not isinstance(
                        corpus, bm25s.utils.corpus.JsonlCorpus
                    ):
                        self.assertIs(documents[position], expected_document)
                if return_as == "tuple":
                    np.testing.assert_array_equal(result.scores, self.expected.scores)

    def test_retrieve_preserves_corpus_items(self):
        corpora = [
            [["cat", "first"], ["dog", "second"], ["fish", "third"]],
            [("cat", 1), ("dog", 2), ("fish", 3)],
            [["cat"], ["dog", "second"], ["fish", "third", 3]],
            [[], [], []],
            [1, "second", 3],
            [{"text": "cat"}, {"text": "dog"}, {"text": "fish"}],
            ["cat", "dog", "fish"],
            [1, 2, 3],
            np.array(["cat", "dog", "fish"]),
            np.array([1, 2, 3], dtype=np.int32),
        ]
        for corpus in corpora:
            with self.subTest(corpus=corpus):
                self.assert_corpus_results(self.retriever, corpus)

    def test_retrieve_preserves_saved_list_items(self):
        corpus = [["cat", 1], ["dog", 2], ["fish", 3]]
        with tempfile.TemporaryDirectory() as path:
            self.retriever.save(path, corpus=corpus, show_progress=False)
            for mmap in (False, True):
                with self.subTest(mmap=mmap):
                    loaded = bm25s.BM25.load(
                        path,
                        load_corpus=True,
                        mmap=mmap,
                        show_progress=False,
                        auto_compile=False,
                    )
                    try:
                        self.assert_corpus_results(loaded, loaded.corpus)
                    finally:
                        if mmap:
                            loaded.corpus.close()


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

    def test_retrieve_with_different_return_types(self):
        queries = [
            "a cat is a feline, it's sometimes beautiful but cannot fly",
            "a dog is the human's best friend and loves to play"
        ]
        for method in ['bm25+', 'lucene', 'bm25l', 'atire', 'robertson']:
            all_docs = []
            all_scores = []
            for return_type in ['ids', 'tuple', 'string']:
                tokenizer = bm25s.tokenization.Tokenizer(lower=True,stopwords="en", stemmer=self.stemmer)
                corpus_tokens = tokenizer.tokenize(self.corpus, return_as=return_type, show_progress=False, allow_empty=True)
                query_tokens = tokenizer.tokenize(queries, return_as=return_type, show_progress=False, allow_empty=True)
                # Create the BM25 model and index the corpus
                retriever = bm25s.BM25(method=method)
                retriever.index(corpus_tokens)

                docs, scores = retriever.retrieve(query_tokens, k=2, sorted=False)
                all_docs.append(docs)
                all_scores.append(scores)
            
            # Check if the results are the same for both return types
            for doc in all_docs[1:]:
                self.assertTrue(np.array_equal(all_docs[0], doc), f"Expected {all_docs[0]}, got {doc}")
            # Check if the scores are the same for both return types
            for score in all_scores[1:]:
                self.assertTrue(np.array_equal(all_scores[0], score), f"Expected {all_scores[0]}, got {score}")


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
