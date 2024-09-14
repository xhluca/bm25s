from typing import Generator
import unittest
import Stemmer
import re

from bm25s.tokenization import Tokenizer

class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define a sample corpus
        cls.corpus = [
            "This is a test sentence.",
            "Another sentence for testing.",
            "Machine learning is fun!",
            "The quick brown fox jumps over the lazy dog.",
        ]

        cls.corpus_large = []

        # load tests/data/nfcorpus.txt
        with open("tests/data/nfcorpus.txt", "r") as file:
            for line in file:
                cls.corpus_large.append(line.strip())

        # Initialize a stemmer
        cls.stemmer = Stemmer.Stemmer("english")

    def setUp(self):
        # Initialize the Tokenizer with default settings
        self.tokenizer = Tokenizer(stemmer=self.stemmer)

    def test_tokenize_with_default_settings(self):
        """Tests the `tokenize` method with default settings."""
        result = self.tokenizer.tokenize(self.corpus, update_vocab=True, return_as="ids", show_progress=True)
        self.assertIsInstance(result, list)
        for doc in result:
            self.assertIsInstance(doc, list)
            for token_id in doc:
                self.assertIsInstance(token_id, int)

    def test_tokenize_with_custom_splitter(self):
        """Tests the `tokenize` method and `__init__` method with a custom splitter."""
        custom_splitter = lambda text: re.findall(r"\w+", text)
        tokenizer = Tokenizer(splitter=custom_splitter, stemmer=self.stemmer)
        result = tokenizer.tokenize(self.corpus, update_vocab=True, return_as="ids", show_progress=False)
        self.assertIsInstance(result, list)

    def test_tokenize_with_stopwords(self):
        """Tests the `tokenize` method and `__init__` method with stopwords filtering."""
        stopwords = ["is", "a", "for"]
        tokenizer = Tokenizer(stopwords=stopwords, stemmer=self.stemmer)
        result = tokenizer.tokenize(self.corpus, update_vocab=True, return_as="string", show_progress=False)
        for doc in result:
            for token in doc:
                self.assertNotIn(token, stopwords)

    def test_tokenize_with_never_update_vocab(self):
        """Tests the `tokenize` method with the `update_vocab="never"` parameter."""
        tokenizer = Tokenizer(stemmer=self.stemmer)
        tokenizer.tokenize(self.corpus, update_vocab="never", show_progress=False)
        vocab_size = len(tokenizer.get_vocab_dict())
        self.assertEqual(vocab_size, 0)

    def test_invalid_splitter(self):
        """Tests the `__init__` method for handling an invalid `splitter` input."""
        with self.assertRaises(ValueError):
            Tokenizer(splitter=123)

    def test_invalid_stemmer(self):
        """Tests the `__init__` method for handling an invalid `stemmer` input."""
        with self.assertRaises(ValueError):
            Tokenizer(stemmer="not_callable")

    def test_tokenize_with_empty_vocab(self):
        """Tests the `tokenize` method with the `update_vocab="if_empty"` parameter."""
        tokenizer = Tokenizer(stemmer=self.stemmer)
        tokenizer.tokenize(self.corpus, update_vocab="if_empty", show_progress=False)
        vocab_size = len(tokenizer.get_vocab_dict())
        self.assertGreater(vocab_size, 0)

    def test_streaming_tokenize(self):
        """Tests the `streaming_tokenize` method directly for its functionality."""
        stream = self.tokenizer.streaming_tokenize(self.corpus)
        assert isinstance(stream, Generator)
        for doc_ids in stream:
            self.assertIsInstance(doc_ids, list)
            for token_id in doc_ids:
                self.assertIsInstance(token_id, int)

    def test_get_vocab_dict(self):
        """Tests the `get_vocab_dict` method to ensure it returns the correct vocabulary dictionary."""
        self.tokenizer.tokenize(self.corpus, update_vocab=True, show_progress=False)
        vocab = self.tokenizer.get_vocab_dict()
        self.assertIsInstance(vocab, dict)
        self.assertGreater(len(vocab), 0)

    def test_tokenize_return_types(self):
        """Tests the `tokenize` method with different `return_as` parameter values (`ids`, `string`, `tuple`)."""
        result_ids = self.tokenizer.tokenize(self.corpus, return_as="ids", show_progress=False)
        result_strings = self.tokenizer.tokenize(self.corpus, return_as="string", show_progress=False)
        result_tuple = self.tokenizer.tokenize(self.corpus, return_as="tuple", show_progress=False)

        self.assertIsInstance(result_ids, list)
        self.assertIsInstance(result_strings, list)
        self.assertIsInstance(result_tuple, tuple)

    def test_tokenize_with_invalid_return_type(self):
        """Tests the `tokenize` method for handling an invalid `return_as` parameter value."""
        with self.assertRaises(ValueError):
            self.tokenizer.tokenize(self.corpus, return_as="invalid_type")

    def test_reset_vocab(self):
        """Tests the `reset_vocab` method to ensure it properly clears all vocabulary dictionaries."""
        self.tokenizer.tokenize(self.corpus, update_vocab=True, show_progress=False)
        self.tokenizer.reset_vocab()
        vocab = self.tokenizer.get_vocab_dict()
        self.assertEqual(len(vocab), 0)

    def test_to_tokenized_tuple(self):
        """Tests the `to_tokenized_tuple` method to ensure it correctly converts token IDs to a named tuple."""
        docs = self.tokenizer.tokenize(self.corpus, return_as="ids", show_progress=False)
        tokenized_tuple = self.tokenizer.to_tokenized_tuple(docs)
        self.assertIsInstance(tokenized_tuple, tuple)
        self.assertEqual(len(tokenized_tuple.ids), len(docs))

    def test_decode_method(self):
        """Tests the `to_lists_of_strings` method to ensure it converts token IDs back to strings properly."""
        docs = self.tokenizer.tokenize(self.corpus_large[:1000], return_as="ids", show_progress=False)
        strings = self.tokenizer.decode(docs)
        self.assertIsInstance(strings, list)
        for doc in strings:
            self.assertIsInstance(doc, list)
            for token in doc:
                self.assertIsInstance(token, str)

    # compare return_ids with decode
    def test_compare_class_with_functional(self):
        """Tests the `to_lists_of_strings` method to ensure it converts token IDs back to strings properly."""
        docs = self.tokenizer.tokenize(self.corpus_large, return_as="ids", show_progress=False)
        strings = self.tokenizer.decode(docs)

        # now, do the same using bm25s.tokenize
        strings2 = self.tokenizer.tokenize(self.corpus_large, return_as="string")
        
        # compare the two
        self.assertEqual(strings, strings2)
    
    @classmethod
    def tearDownClass(cls):
        """Cleans up resources after all tests have run (not required in this test case)."""
        pass


if __name__ == "__main__":
    unittest.main()
