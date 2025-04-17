import unittest
import numpy as np
import bm25s


class TestBM25SAllowEmpty(unittest.TestCase):
    def test_allow_empty_parameter(self):
        # Test corpus with some entries that might be empty after tokenization
        corpus = ['foo', 'dog', 'baz', 'quick god', 'quick fox']
        query = 'quick'
        
        # First test with allow_empty=True (should keep empty documents)
        tokenizer_allow_empty = bm25s.tokenization.Tokenizer(stopwords=["english"])
        corpus_tokens_with_empty = tokenizer_allow_empty.tokenize(
            corpus, 
            show_progress=False, 
            allow_empty=True
        )
        
        # Then test with allow_empty=False (should filter out empty documents)
        tokenizer_no_empty = bm25s.tokenization.Tokenizer(stopwords=["english"])
        corpus_tokens_no_empty = tokenizer_no_empty.tokenize(
            corpus,
            show_progress=False,
            allow_empty=False
        )
        
        # Verify that the two tokenizations are different
        self.assertNotEqual(
            len(corpus_tokens_with_empty[0]), 
            len(corpus_tokens_no_empty[0]),
            "allow_empty=True and allow_empty=False should produce different results"
        )
        
        # Test BM25 indexing and retrieval with allow_empty=True
        retriever_with_empty = bm25s.BM25(backend="numpy")
        retriever_with_empty.index(corpus_tokens_with_empty, show_progress=False)
        query_tokens_with_empty = tokenizer_allow_empty.tokenize(
            [query], 
            show_progress=False, 
            allow_empty=True
        )
        results_with_empty, scores_with_empty = retriever_with_empty.retrieve(
            query_tokens_with_empty, 
            k=len(corpus), 
            show_progress=False, 
            n_threads=1, 
            sorted=True
        )
        
        # Test BM25 indexing and retrieval with allow_empty=False
        retriever_no_empty = bm25s.BM25(backend="numpy")
        retriever_no_empty.index(corpus_tokens_no_empty, show_progress=False)
        query_tokens_no_empty = tokenizer_no_empty.tokenize(
            [query], 
            show_progress=False, 
            allow_empty=False
        )
        results_no_empty, scores_no_empty = retriever_no_empty.retrieve(
            query_tokens_no_empty, 
            k=len(corpus_tokens_no_empty[0]), 
            show_progress=False, 
            n_threads=1, 
            sorted=True
        )
        
        # Verify that the results are different between the two approaches
        self.assertNotEqual(
            results_with_empty.shape, 
            results_no_empty.shape,
            "Result shapes should differ between allow_empty=True and allow_empty=False"
        )
        
        # When allow_empty=False, we should only retrieve documents that contain 'quick'
        for idx in results_no_empty[0]:
            # Get the original document text from the filtered corpus
            # (need to map back to original corpus indices)
            doc_text = corpus[int(idx)]
            self.assertIn(
                'quick', 
                doc_text, 
                f"Document at index {idx} should contain 'quick' when allow_empty=False"
            )
        
        # For both cases, verify that the top results are documents with 'quick'
        self.assertIn(
            3, 
            results_with_empty[0][:2], 
            "Document with 'quick god' should be in top results when allow_empty=True"
        )
        self.assertIn(
            4, 
            results_with_empty[0][:2], 
            "Document with 'quick fox' should be in top results when allow_empty=True"
        )
        
        # The document indices in results_no_empty might be different due to filtering
        # But the top documents should still contain 'quick'
        for idx in results_no_empty[0][:2]:
            doc_text = corpus[int(idx)]
            self.assertIn(
                'quick', 
                doc_text, 
                "Top results should contain 'quick' when allow_empty=False"
            )
            
    def test_empty_document_handling(self):
        # Test with a corpus where some documents become empty after tokenization
        corpus = ['the of and', 'dog', 'in at', 'quick god', 'quick fox']
        query = 'quick'
        
        # With allow_empty=True, we should keep all documents
        tokenizer = bm25s.tokenization.Tokenizer(stopwords=["english"])
        corpus_tokens_with_empty = tokenizer.tokenize(
            corpus, 
            show_progress=False, 
            allow_empty=True
        )
        
        # With allow_empty=False, we should filter out empty documents
        corpus_tokens_no_empty = tokenizer.tokenize(
            corpus, 
            show_progress=False, 
            allow_empty=False
        )
        
        # Verify that allow_empty=True keeps all documents
        self.assertEqual(
            len(corpus_tokens_with_empty[0]), 
            len(corpus),
            "With allow_empty=True, all documents should be kept"
        )
        
        # Verify that allow_empty=False filters out empty documents
        self.assertLess(
            len(corpus_tokens_no_empty[0]), 
            len(corpus),
            "With allow_empty=False, empty documents should be filtered out"
        )
        
        # BM25 retrieval with allow_empty=False should only include non-empty documents
        retriever_no_empty = bm25s.BM25(backend="numpy")
        retriever_no_empty.index(corpus_tokens_no_empty, show_progress=False)
        query_tokens = tokenizer.tokenize(
            [query], 
            show_progress=False, 
            allow_empty=False
        )
        
        results, scores = retriever_no_empty.retrieve(
            query_tokens, 
            k=len(corpus_tokens_no_empty[0]), 
            show_progress=False, 
            n_threads=1, 
            sorted=True
        )
        
        # Check that the only results returned are documents containing 'quick'
        # (after filtering out empty documents)
        for idx in results[0]:
            original_idx = corpus_tokens_no_empty[0][int(idx)]
            doc_text = corpus[original_idx]
            self.assertIn(
                'quick', 
                doc_text, 
                f"Document at index {original_idx} should contain 'quick'"
            )


if __name__ == "__main__":
    unittest.main()