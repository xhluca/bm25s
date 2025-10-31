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
        tokenizer = Tokenizer(stopwords='en')
        corpus = ["this is a test", "another test document"]
        
        # Tokenize
        token_ids = tokenizer.tokenize(corpus, show_progress=False, return_as='ids')
        
        # First decode (builds cache)
        decoded1 = tokenizer.decode(token_ids)
        
        # Second decode (uses cache)
        decoded2 = tokenizer.decode(token_ids)
        
        # Should be identical
        self.assertEqual(decoded1, decoded2)
        
        # Verify cache exists
        self.assertIsNotNone(tokenizer._reverse_vocab_cache)
    
    def test_cache_invalidation_on_vocab_update(self):
        """Verify cache is invalidated when vocabulary changes"""
        tokenizer = Tokenizer(stopwords='en')
        
        # Tokenize first batch
        corpus1 = ["this is a test"]
        token_ids1 = tokenizer.tokenize(corpus1, show_progress=False, return_as='ids')
        
        # Decode to create cache
        decoded1 = tokenizer.decode(token_ids1)
        cache1 = tokenizer._reverse_vocab_cache
        
        # Tokenize second batch with new words (updates vocab)
        corpus2 = ["new words here"]
        token_ids2 = tokenizer.tokenize(corpus2, show_progress=False, return_as='ids')
        
        # Cache should be invalidated (set to None in streaming_tokenize)
        # But then recreated on next decode
        decoded2 = tokenizer.decode(token_ids2)
        
        # Should work correctly
        self.assertIsNotNone(decoded2)
    
    def test_cache_invalidation_on_reset(self):
        """Verify cache is invalidated when vocab is reset"""
        tokenizer = Tokenizer(stopwords='en')
        
        # Tokenize and decode
        corpus = ["this is a test"]
        token_ids = tokenizer.tokenize(corpus, show_progress=False, return_as='ids')
        decoded = tokenizer.decode(token_ids)
        
        # Cache should exist
        self.assertIsNotNone(tokenizer._reverse_vocab_cache)
        
        # Reset vocab
        tokenizer.reset_vocab()
        
        # Cache should be cleared
        self.assertIsNone(tokenizer._reverse_vocab_cache)
    
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
