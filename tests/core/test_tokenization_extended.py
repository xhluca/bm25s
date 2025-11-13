import unittest
import bm25s
from bm25s.tokenization import Tokenizer, _infer_stopwords


class TestTokenizationEdgeCases(unittest.TestCase):
    """Test coverage for tokenization edge cases"""

    def test_infer_stopwords_various_languages(self):
        """Test stopwords inference for all supported languages"""
        # Test various language codes
        lang_codes = [
            ("german", "de"),
            ("dutch", "nl"),
            ("french", "fr"),
            ("spanish", "es"),
            ("portuguese", "pt"),
            ("italian", "it"),
            ("russian", "ru"),
            ("swedish", "sv"),
            ("norwegian", "no"),
            ("chinese", "zh"),
            ("turkish", "tr"),
        ]
        
        for lang_name, lang_code in lang_codes:
            # Test full name
            stopwords = _infer_stopwords(lang_name)
            self.assertIsInstance(stopwords, (list, tuple))
            self.assertGreater(len(stopwords), 0)
            
            # Test code
            stopwords = _infer_stopwords(lang_code)
            self.assertIsInstance(stopwords, (list, tuple))
            self.assertGreater(len(stopwords), 0)

    def test_infer_stopwords_none_and_false(self):
        """Test stopwords inference returns empty list for None and False"""
        stopwords = _infer_stopwords(None)
        self.assertEqual(stopwords, [])
        
        stopwords = _infer_stopwords(False)
        self.assertEqual(stopwords, [])

    def test_infer_stopwords_invalid_string(self):
        """Test stopwords inference raises error for invalid language string"""
        with self.assertRaises(ValueError):
            _infer_stopwords("invalid_language_code")

    def test_infer_stopwords_custom_list(self):
        """Test stopwords inference with custom list"""
        custom_stopwords = ["the", "a", "an"]
        stopwords = _infer_stopwords(custom_stopwords)
        self.assertEqual(stopwords, custom_stopwords)

    def test_tokenizer_update_vocab_never(self):
        """Test tokenizer with update_vocab='never'"""
        tokenizer = Tokenizer()
        # Pre-populate vocab
        tokenizer.word_to_id = {"hello": 0, "world": 1}
        
        # Tokenize with update_vocab='never'
        texts = ["hello world new"]
        result = tokenizer.tokenize(texts, update_vocab="never", return_as="ids")
        
        # Should only have hello and world, not 'new'
        self.assertEqual(len(result[0]), 2)

    def test_tokenizer_update_vocab_if_empty_with_empty_vocab(self):
        """Test tokenizer with update_vocab='if_empty' when vocab is empty"""
        tokenizer = Tokenizer()
        
        texts = ["hello world"]
        tokenizer.tokenize(texts, update_vocab="if_empty", return_as="ids")
        
        # Should have updated vocab
        self.assertGreater(len(tokenizer.word_to_id), 0)

    def test_tokenizer_update_vocab_if_empty_with_existing_vocab(self):
        """Test tokenizer with update_vocab='if_empty' when vocab exists"""
        tokenizer = Tokenizer()
        # Pre-populate vocab
        tokenizer.word_to_id = {"hello": 0}
        
        texts = ["hello world"]
        tokenizer.tokenize(texts, update_vocab="if_empty", return_as="ids")
        
        # Should NOT have updated vocab with 'world'
        self.assertEqual(len(tokenizer.word_to_id), 1)

    def test_tokenizer_invalid_update_vocab(self):
        """Test tokenizer raises error for invalid update_vocab value"""
        tokenizer = Tokenizer()
        
        with self.assertRaises(ValueError):
            tokenizer.tokenize(["test"], update_vocab="invalid")

    def test_tokenizer_invalid_return_as(self):
        """Test tokenizer raises error for invalid return_as value"""
        tokenizer = Tokenizer()
        
        with self.assertRaises(ValueError):
            tokenizer.tokenize(["test"], return_as="invalid")

    def test_tokenizer_return_as_stream(self):
        """Test tokenizer with return_as='stream'"""
        tokenizer = Tokenizer()
        
        texts = ["hello world", "foo bar"]
        stream = tokenizer.tokenize(texts, return_as="stream")
        
        # Should return a generator/iterator
        results = list(stream)
        self.assertEqual(len(results), 2)

    def test_tokenizer_with_length_parameter(self):
        """Test tokenizer with explicit length parameter"""
        tokenizer = Tokenizer()
        
        # Create a generator
        texts = (t for t in ["hello world", "foo bar"])
        result = tokenizer.tokenize(texts, length=2, return_as="ids", show_progress=False)
        
        self.assertEqual(len(result), 2)

    def test_tokenize_with_stemmer_callable(self):
        """Test tokenization with callable stemmer"""
        # Create a simple callable stemmer
        def simple_stemmer(tokens):
            return [t[:3] for t in tokens]  # Take up to the first 3 chars (returns whole token if shorter)
        
        texts = ["running jumping"]
        result = bm25s.tokenize(texts, stemmer=simple_stemmer, return_ids=False)
        
        # Tokens should be stemmed
        self.assertIsNotNone(result)

    def test_tokenize_with_invalid_stemmer(self):
        """Test tokenization with invalid stemmer raises error"""
        texts = ["hello world"]
        
        # Pass a non-callable, non-stemmer object
        with self.assertRaises(ValueError) as context:
            bm25s.tokenize(texts, stemmer="not_a_stemmer", return_ids=False)
        self.assertIn("Stemmer must have", str(context.exception))

    def test_tokenize_allow_empty_false(self):
        """Test tokenization with allow_empty=False"""
        texts = [""]  # Empty string
        result = bm25s.tokenize(texts, allow_empty=False, return_ids=True)
        
        # Should have one token (empty string token)
        self.assertEqual(len(result.ids[0]), 1)

    def test_tokenize_allow_empty_true(self):
        """Test tokenization with allow_empty=True"""
        texts = [""]  # Empty string
        result = bm25s.tokenize(texts, allow_empty=True, return_ids=True)
        
        # Should have empty list
        self.assertEqual(len(result.ids[0]), 0)


if __name__ == "__main__":
    unittest.main()
