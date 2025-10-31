"""
Test suite to verify performance optimizations work correctly
"""
import unittest
import bm25s
from bm25s.tokenization import Tokenizer


class TestPerformanceOptimizations(unittest.TestCase):
    
    def test_stopwords_set_created_once(self):
        """Verify stopwords set is created outside the loop"""
        # This is more of a code inspection test, but we can verify
        # it works correctly with many documents
        corpus = [f"the cat sat on the mat number {i}" for i in range(100)]
        result = bm25s.tokenize(corpus, stopwords='en', show_progress=False)
        
        # All documents should have stopwords removed
        for doc_ids in result.ids:
            # Reconstruct tokens to check
            vocab_reverse = {v: k for k, v in result.vocab.items()}
            tokens = [vocab_reverse[id] for id in doc_ids]
            # Verify no stopwords in result
            self.assertNotIn('the', tokens)
            self.assertNotIn('on', tokens)
    
    def test_allow_empty_logic_correct(self):
        """Verify allow_empty=True adds empty token, allow_empty=False does not"""
        # Test with allow_empty=True (should add empty for empty docs)
        corpus_with_empty = ["", "test"]
        result = bm25s.tokenize(corpus_with_empty, stopwords=None, 
                               show_progress=False, allow_empty=True)
        # First doc should have empty token
        self.assertEqual(len(result.ids[0]), 1)
        
        # Test with allow_empty=False (empty docs stay empty)
        result = bm25s.tokenize(corpus_with_empty, stopwords=None,
                               show_progress=False, allow_empty=False)
        # First doc should be empty list
        self.assertEqual(len(result.ids[0]), 0)
    
    def test_decode_caching_works(self):
        """Verify decode caching provides correct results"""
        import time
        tokenizer = Tokenizer(stopwords='en')
        # Use larger vocab to make timing difference more noticeable
        corpus = [f"word{i} test document" for i in range(100)]
        
        # Tokenize
        token_ids = tokenizer.tokenize(corpus, show_progress=False, return_as='ids')
        
        # First decode (builds cache)
        decoded1 = tokenizer.decode(token_ids)
        
        # Second decode (uses cache)
        decoded2 = tokenizer.decode(token_ids)
        
        # Should be identical (behavioral test)
        self.assertEqual(decoded1, decoded2)
        
        # Third decode should also be consistent
        decoded3 = tokenizer.decode(token_ids)
        self.assertEqual(decoded1, decoded3)
    
    def test_cache_invalidation_on_vocab_update(self):
        """Verify cache works correctly when vocabulary changes"""
        tokenizer = Tokenizer(stopwords='en')
        
        # Tokenize first batch
        corpus1 = ["this is a test"]
        token_ids1 = tokenizer.tokenize(corpus1, show_progress=False, return_as='ids', update_vocab=True)
        
        # Decode to create cache
        decoded1 = tokenizer.decode(token_ids1)
        
        # Tokenize second batch with new words (updates vocab because update_vocab=True)
        corpus2 = ["new words here"]
        token_ids2 = tokenizer.tokenize(corpus2, show_progress=False, return_as='ids', update_vocab=True)
        
        # Decode should work correctly with updated vocabulary
        decoded2 = tokenizer.decode(token_ids2)
        
        # Both should decode correctly
        self.assertIsNotNone(decoded1)
        self.assertIsNotNone(decoded2)
        
        # Verify they contain the expected tokens
        self.assertIn('test', decoded1[0])
        self.assertIn('new', decoded2[0])
        self.assertIn('words', decoded2[0])
    
    def test_cache_invalidation_on_reset(self):
        """Verify decode works correctly after vocab reset"""
        tokenizer = Tokenizer(stopwords='en')
        
        # Tokenize and decode
        corpus = ["this is a test"]
        token_ids = tokenizer.tokenize(corpus, show_progress=False, return_as='ids')
        decoded = tokenizer.decode(token_ids)
        
        # Verify decode worked
        self.assertIsNotNone(decoded)
        
        # Reset vocab
        tokenizer.reset_vocab()
        
        # Tokenize new corpus after reset
        corpus2 = ["new corpus after reset"]
        token_ids2 = tokenizer.tokenize(corpus2, show_progress=False, return_as='ids')
        decoded2 = tokenizer.decode(token_ids2)
        
        # Should work correctly with new vocabulary
        self.assertIsNotNone(decoded2)
        self.assertIn('new', decoded2[0])
        self.assertIn('corpus', decoded2[0])
    
    def test_doc_freq_calculation_correctness(self):
        """Verify document frequency calculation is correct with optimization"""
        from bm25s.scoring import _calculate_doc_freqs
        
        # Create test corpus
        corpus_tokens = [
            [0, 1, 2],      # doc 0: tokens 0, 1, 2
            [1, 2, 3],      # doc 1: tokens 1, 2, 3
            [0, 2, 4],      # doc 2: tokens 0, 2, 4
        ]
        unique_tokens = [0, 1, 2, 3, 4]
        
        # Calculate doc frequencies
        doc_freqs = _calculate_doc_freqs(corpus_tokens, unique_tokens, show_progress=False)
        
        # Verify counts
        self.assertEqual(doc_freqs[0], 2)  # token 0 in docs 0, 2
        self.assertEqual(doc_freqs[1], 2)  # token 1 in docs 0, 1
        self.assertEqual(doc_freqs[2], 3)  # token 2 in docs 0, 1, 2
        self.assertEqual(doc_freqs[3], 1)  # token 3 in doc 1
        self.assertEqual(doc_freqs[4], 1)  # token 4 in doc 2


if __name__ == '__main__':
    unittest.main()
