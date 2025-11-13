import unittest
import numpy as np
from bm25s.scoring import (
    _select_tfc_scorer,
    _select_idf_scorer,
    _compute_relevance_from_scores_legacy,
    _compute_relevance_from_scores_jit_ready,
)


class TestScoringFunctions(unittest.TestCase):
    """Test coverage for scoring.py functions"""

    def test_select_tfc_scorer_invalid_method(self):
        """Test that _select_tfc_scorer raises ValueError for invalid method"""
        with self.assertRaises(ValueError) as context:
            _select_tfc_scorer("invalid_method")
        self.assertIn("Invalid score_tfc value", str(context.exception))

    def test_select_idf_scorer_invalid_method(self):
        """Test that _select_idf_scorer raises ValueError for invalid method"""
        with self.assertRaises(ValueError) as context:
            _select_idf_scorer("invalid_method")
        self.assertIn("Invalid score_idf_inner value", str(context.exception))

    def test_compute_relevance_from_scores_legacy(self):
        """Test the legacy implementation of _compute_relevance_from_scores"""
        # Create sample data for testing
        # Simple case: 3 docs, 2 tokens
        # Token 0 appears in doc 0 with score 1.0
        # Token 1 appears in doc 1 with score 2.0
        data = np.array([1.0, 2.0], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        num_docs = 3
        query_tokens_ids = [0, 1]
        dtype = np.float32

        scores = _compute_relevance_from_scores_legacy(
            data, indptr, indices, num_docs, query_tokens_ids, dtype
        )

        # Expected: doc 0 has score 1.0, doc 1 has score 2.0, doc 2 has score 0.0
        expected = np.array([1.0, 2.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(scores, expected)

    def test_compute_relevance_from_scores_legacy_empty_query(self):
        """Test the legacy implementation with empty query"""
        data = np.array([1.0, 2.0], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        num_docs = 3
        query_tokens_ids = []
        dtype = np.float32

        scores = _compute_relevance_from_scores_legacy(
            data, indptr, indices, num_docs, query_tokens_ids, dtype
        )

        # Expected: all zeros for empty query
        expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(scores, expected)

    def test_compute_relevance_from_scores_jit_ready(self):
        """Test the JIT-ready implementation of _compute_relevance_from_scores"""
        # Create sample data for testing
        data = np.array([1.0, 2.0], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        num_docs = 3
        query_tokens_ids = np.array([0, 1], dtype=np.int32)
        dtype = np.float32

        scores = _compute_relevance_from_scores_jit_ready(
            data, indptr, indices, num_docs, query_tokens_ids, dtype
        )

        # Expected: doc 0 has score 1.0, doc 1 has score 2.0, doc 2 has score 0.0
        expected = np.array([1.0, 2.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(scores, expected)

    def test_compute_relevance_from_scores_jit_ready_multiple_tokens_same_doc(self):
        """Test the JIT-ready implementation with multiple tokens in same doc"""
        # Create sample data: both tokens appear in doc 0
        data = np.array([1.5, 2.5], dtype=np.float32)
        indices = np.array([0, 0], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        num_docs = 2
        query_tokens_ids = np.array([0, 1], dtype=np.int32)
        dtype = np.float32

        scores = _compute_relevance_from_scores_jit_ready(
            data, indptr, indices, num_docs, query_tokens_ids, dtype
        )

        # Expected: doc 0 has score 1.5 + 2.5 = 4.0, doc 1 has score 0.0
        expected = np.array([4.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(scores, expected)


if __name__ == "__main__":
    unittest.main()
