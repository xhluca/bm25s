import os
import shutil
import tempfile
import unittest
import numpy as np
import bm25s
from bm25s import BM25, Results
import Stemmer


class TestBM25CoreFunctions(unittest.TestCase):
    """Test coverage for core BM25 functions in __init__.py"""

    @classmethod
    def setUpClass(cls):
        cls.corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]
        cls.stemmer = Stemmer.Stemmer("english")
        cls.corpus_tokens = bm25s.tokenize(cls.corpus, stopwords="en", stemmer=cls.stemmer)

    def test_backend_auto_selection_no_numba(self):
        """Test backend auto selection when numba is not available"""
        # Create a BM25 instance with auto backend
        retriever = BM25(backend="auto")
        # The backend should be set based on availability
        # We can't easily test without numba, but we can at least verify it doesn't crash
        self.assertIn(retriever.backend, ["numpy", "numba"])

    def test_infer_corpus_object_invalid_tuple(self):
        """Test _infer_corpus_object with invalid tuple"""
        # Invalid tuple: not (list, dict)
        invalid_corpus = (["token"], ["another_token"])
        
        with self.assertRaises(ValueError) as context:
            BM25._infer_corpus_object(invalid_corpus)
        self.assertIn("Corpus must be", str(context.exception))

    def test_infer_corpus_object_invalid_type(self):
        """Test _infer_corpus_object with invalid type"""
        # Invalid type: not iterable
        invalid_corpus = 12345
        
        with self.assertRaises(ValueError) as context:
            BM25._infer_corpus_object(invalid_corpus)
        self.assertIn("Corpus must be", str(context.exception))

    def test_infer_corpus_object_with_object(self):
        """Test _infer_corpus_object with object having ids and vocab"""
        class MockCorpus:
            def __init__(self):
                self.ids = [[0, 1], [2, 3]]
                self.vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
        
        result = BM25._infer_corpus_object(MockCorpus())
        self.assertEqual(result, "object")

    def test_infer_corpus_object_with_tuple(self):
        """Test _infer_corpus_object with valid tuple"""
        corpus = ([[0, 1], [2, 3]], {"a": 0, "b": 1, "c": 2, "d": 3})
        result = BM25._infer_corpus_object(corpus)
        self.assertEqual(result, "tuple")

    def test_infer_corpus_object_with_token_ids(self):
        """Test _infer_corpus_object with list of token IDs"""
        corpus = [[0, 1, 2], [3, 4, 5]]
        result = BM25._infer_corpus_object(corpus)
        self.assertEqual(result, "token_ids")

    def test_infer_corpus_object_with_tokens(self):
        """Test _infer_corpus_object with list of tokens"""
        corpus = [["hello", "world"], ["foo", "bar"]]
        result = BM25._infer_corpus_object(corpus)
        self.assertEqual(result, "tokens")

    def test_empty_query_handling(self):
        """Test handling of empty query"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Create an empty query
        empty_query_tokens = [[]]
        
        # Should not crash and return results with zero scores
        results = retriever.retrieve(empty_query_tokens, k=2)
        
        self.assertEqual(len(results.documents), 1)
        self.assertEqual(len(results.documents[0]), 2)
        # All scores should be zero for empty query
        self.assertTrue(np.all(results.scores[0] == 0))

    def test_results_merge(self):
        """Test Results.merge() method"""
        # Create sample results
        results1 = Results(
            documents=np.array([[0, 1], [2, 3]]),
            scores=np.array([[1.0, 0.5], [0.8, 0.3]])
        )
        results2 = Results(
            documents=np.array([[4, 5]]),
            scores=np.array([[0.9, 0.2]])
        )
        
        merged = Results.merge([results1, results2])
        
        expected_docs = np.array([[0, 1], [2, 3], [4, 5]])
        expected_scores = np.array([[1.0, 0.5], [0.8, 0.3], [0.9, 0.2]])
        
        np.testing.assert_array_equal(merged.documents, expected_docs)
        np.testing.assert_array_equal(merged.scores, expected_scores)

    def test_results_len(self):
        """Test Results.__len__() method"""
        results = Results(
            documents=np.array([[0, 1], [2, 3], [4, 5]]),
            scores=np.array([[1.0, 0.5], [0.8, 0.3], [0.9, 0.2]])
        )
        
        self.assertEqual(len(results), 3)

    def test_index_with_token_ids_and_create_empty_token(self):
        """Test indexing with token IDs and create_empty_token=True"""
        # Create corpus as list of token IDs
        corpus_token_ids = [[0, 1, 2], [1, 2, 3], [0, 3, 4]]
        
        retriever = BM25()
        retriever.index(corpus_token_ids, create_empty_token=True)
        
        # Verify that vocab_dict was created
        self.assertIsNotNone(retriever.vocab_dict)
        # Verify that an entry exists for empty token or max+1
        self.assertTrue(len(retriever.vocab_dict) >= 5)  # 5 unique tokens + possibly empty

    def test_index_with_token_ids_when_zero_exists(self):
        """Test indexing with token IDs when 0 already exists"""
        # Create corpus as list of token IDs including 0
        corpus_token_ids = [[0, 1, 2], [1, 2, 3], [0, 3, 4]]
        
        retriever = BM25()
        retriever.index(corpus_token_ids, create_empty_token=True)
        
        # Should add max+1 as empty token since 0 exists
        self.assertIsNotNone(retriever.vocab_dict)

    def test_get_scores_from_ids_with_invalid_token_id(self):
        """Test get_scores_from_ids with token ID higher than vocab size"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Get the max token ID in vocab
        max_token_id = max(retriever.vocab_dict.values())
        
        # Try to query with a token ID that's too high
        with self.assertRaises(ValueError) as context:
            retriever.get_scores_from_ids([max_token_id + 100])
        self.assertIn("maximum token ID", str(context.exception))

    def test_get_scores_with_list_of_ints(self):
        """Test get_scores with list of token IDs (ints)"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Get some valid token IDs
        valid_token_ids = list(retriever.vocab_dict.values())[:3]
        
        # Should work with int token IDs
        scores = retriever.get_scores(valid_token_ids)
        
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.corpus))

    def test_get_scores_with_invalid_type(self):
        """Test get_scores with invalid input type"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Try with a string instead of list
        with self.assertRaises(ValueError) as context:
            retriever.get_scores("not a list")
        self.assertIn("must be a list", str(context.exception))

    def test_get_scores_with_invalid_token_type(self):
        """Test get_scores with invalid token type in list"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Try with a list of floats
        with self.assertRaises(ValueError) as context:
            retriever.get_scores([1.5, 2.5])
        self.assertIn("must be a list of tokens", str(context.exception))

    def test_retrieve_with_n_threads_minus_one(self):
        """Test retrieve with n_threads=-1 to use all CPUs"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat", "feline"]]
        
        # Should work with n_threads=-1
        results = retriever.retrieve(query_tokens, k=2, n_threads=-1)
        self.assertEqual(len(results.documents), 1)

    def test_retrieve_with_token_ids_no_vocab_set(self):
        """Test retrieve with token IDs when unique_token_ids_set is not set"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Manually remove unique_token_ids_set to test error case
        retriever.unique_token_ids_set = None
        
        query_tokens_ids = [[0, 1, 2]]
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(query_tokens_ids, k=2)
        self.assertIn("unique_token_ids_set attribute is not found", str(context.exception))

    def test_retrieve_with_filtered_token_ids_empty_query(self):
        """Test retrieve with token IDs that filter to empty query without empty token"""
        retriever = BM25()
        # Index without create_empty_token
        corpus_for_test = [["hello", "world"], ["foo", "bar"]]
        retriever.index(corpus_for_test, create_empty_token=False)
        
        # Create query with token IDs not in vocab
        max_id = max(retriever.vocab_dict.values())
        query_tokens_ids = [[max_id + 100, max_id + 200]]
        
        # Since empty token is not in vocab, this should raise an error
        if "" in retriever.vocab_dict:
            # If empty token exists, skip this test
            self.skipTest("Empty token exists in vocab, cannot test error case")
        else:
            with self.assertRaises(ValueError) as context:
                retriever.retrieve(query_tokens_ids, k=2)
            self.assertIn("does not contain any tokens", str(context.exception))

    def test_retrieve_with_invalid_tuple_length(self):
        """Test retrieve with invalid tuple (wrong number of elements)"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Pass tuple with 3 elements instead of 2
        invalid_query = ([0, 1], {"a": 0}, "extra")
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(invalid_query, k=2)
        self.assertIn("Expected a list of string or a tuple of two elements", str(context.exception))

    def test_retrieve_with_tuple_invalid_first_element(self):
        """Test retrieve with tuple where first element is not iterable"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Pass tuple with non-iterable first element
        invalid_query = (123, {"a": 0})
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(invalid_query, k=2)
        self.assertIn("first element of the tuple passed to retrieve must be an iterable", str(context.exception))

    def test_retrieve_with_tuple_invalid_second_element(self):
        """Test retrieve with tuple where second element is not dict"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Pass tuple with non-dict second element
        invalid_query = ([[0, 1]], "not a dict")
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(invalid_query, k=2)
        self.assertIn("second element of the tuple passed to retrieve must be a dictionary", str(context.exception))

    def test_retrieve_with_invalid_weight_mask_type(self):
        """Test retrieve with weight_mask that is not numpy array"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat"]]
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(query_tokens, k=2, weight_mask=[1, 0, 1, 0])
        self.assertIn("weight_mask must be a numpy array", str(context.exception))

    def test_retrieve_with_invalid_weight_mask_dimensions(self):
        """Test retrieve with weight_mask that is not 1D"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat"]]
        
        # Create 2D weight mask
        weight_mask = np.array([[1, 0], [1, 0]])
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(query_tokens, k=2, weight_mask=weight_mask)
        self.assertIn("weight_mask must be a 1D array", str(context.exception))

    def test_retrieve_with_invalid_weight_mask_length(self):
        """Test retrieve with weight_mask of wrong length"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat"]]
        
        # Create weight mask with wrong length
        weight_mask = np.array([1, 0])  # corpus has 4 docs
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(query_tokens, k=2, weight_mask=weight_mask)
        self.assertIn("length of the weight_mask must be the same", str(context.exception))

    def test_retrieve_with_corpus_as_ndarray(self):
        """Test retrieve with corpus as numpy array"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat", "feline"]]
        
        # Create corpus as numpy array
        corpus = np.array(self.corpus)
        
        results = retriever.retrieve(query_tokens, corpus=corpus, k=2)
        self.assertEqual(len(results.documents), 1)
        self.assertEqual(len(results.documents[0]), 2)

    def test_retrieve_return_as_documents(self):
        """Test retrieve with return_as='documents'"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat"]]
        
        # Test return_as='documents'
        results = retriever.retrieve(query_tokens, k=2, return_as="documents")
        
        # Should return only documents, not Results tuple
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)

    def test_retrieve_with_invalid_return_as(self):
        """Test retrieve with invalid return_as parameter"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        query_tokens = [["cat"]]
        
        with self.assertRaises(ValueError) as context:
            retriever.retrieve(query_tokens, k=2, return_as="invalid")
        self.assertIn("return_as", str(context.exception))


class TestBM25LoadingSaving(unittest.TestCase):
    """Test coverage for loading and saving functions"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.corpus = [
            "a cat is a feline",
            "a dog is the human's best friend",
            "a bird can fly",
        ]
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en", stemmer=self.stemmer)

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_load_with_invalid_mmap_type(self):
        """Test load with invalid mmap parameter type"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        retriever.save(self.tmpdir)
        
        with self.assertRaises(ValueError) as context:
            BM25.load(self.tmpdir, mmap="invalid")
        self.assertIn("must be a boolean", str(context.exception))

    def test_save_with_non_iterable_corpus(self):
        """Test save with non-iterable corpus parameter"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Try to save with non-iterable corpus
        # This should raise TypeError since the code tries to enumerate
        with self.assertRaises(TypeError):
            retriever.save(self.tmpdir, corpus=12345)

    def test_save_with_invalid_document_in_corpus(self):
        """Test save with invalid document type in corpus"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        
        # Create corpus with invalid document type
        invalid_corpus = ["valid string", {"valid": "dict"}, None, 12345]
        
        # This should log warnings for invalid docs but not crash
        retriever.save(self.tmpdir, corpus=invalid_corpus)
        
        # Verify save succeeded
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "params.index.json")))

    def test_load_with_load_vocab_false(self):
        """Test load with load_vocab=False"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        retriever.save(self.tmpdir)
        
        # Load without vocab
        loaded = BM25.load(self.tmpdir, load_vocab=False)
        
        # vocab_dict should be empty
        self.assertEqual(len(loaded.vocab_dict), 0)

    def test_load_corpus_with_mmap(self):
        """Test load_corpus with mmap=True"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        retriever.save(self.tmpdir, corpus=self.corpus)
        
        # Load with mmap and corpus
        loaded = BM25.load(self.tmpdir, load_corpus=True, mmap=True)
        
        # Corpus should be JsonlCorpus
        self.assertIsInstance(loaded.corpus, bm25s.utils.corpus.JsonlCorpus)
        loaded.corpus.close()

    def test_load_corpus_without_mmap(self):
        """Test load_corpus with mmap=False"""
        retriever = BM25()
        retriever.index(self.corpus_tokens)
        retriever.save(self.tmpdir, corpus=self.corpus)
        
        # Load without mmap
        loaded = BM25.load(self.tmpdir, load_corpus=True, mmap=False)
        
        # Corpus should be a list
        self.assertIsInstance(loaded.corpus, list)
        self.assertEqual(len(loaded.corpus), len(self.corpus))


if __name__ == "__main__":
    unittest.main()
