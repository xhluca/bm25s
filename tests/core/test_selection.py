import unittest
import numpy as np
from bm25s.selection import topk


class TestSelectionFunctions(unittest.TestCase):
    """Test coverage for selection.py functions"""

    def test_topk_invalid_backend(self):
        """Test that topk raises ValueError for invalid backend"""
        query_scores = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)
        
        with self.assertRaises(ValueError) as context:
            topk(query_scores, k=2, backend="invalid_backend")
        self.assertIn("Invalid backend", str(context.exception))

    def test_topk_jax_not_available(self):
        """Test that topk raises ImportError when JAX backend is requested but not available"""
        query_scores = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)
        
        # This test will only fail if JAX is not available
        # Since JAX might be installed, we'll use a different approach
        # We'll test the error path by checking if the import error is raised
        try:
            import jax
            # If JAX is available, skip this test or test with numpy backend
            self.skipTest("JAX is available, cannot test error case")
        except ImportError:
            # JAX is not available, test should raise ImportError
            with self.assertRaises(ImportError) as context:
                topk(query_scores, k=2, backend="jax")
            self.assertIn("JAX is not available", str(context.exception))

    def test_topk_numpy_backend(self):
        """Test topk with numpy backend"""
        query_scores = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)
        
        scores, indices = topk(query_scores, k=2, backend="numpy", sorted=True)
        
        # Should return top 2: 4.0 (index 2) and 3.0 (index 0)
        expected_scores = np.array([4.0, 3.0], dtype=np.float32)
        expected_indices = np.array([2, 0], dtype=np.int64)
        
        np.testing.assert_array_almost_equal(scores, expected_scores)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_topk_numpy_backend_unsorted(self):
        """Test topk with numpy backend unsorted"""
        query_scores = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)
        
        scores, indices = topk(query_scores, k=2, backend="numpy", sorted=False)
        
        # Should return top 2 but unsorted
        # Check that we got the right values (order doesn't matter)
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(indices), 2)
        self.assertIn(4.0, scores)
        self.assertIn(3.0, scores)

    def test_topk_auto_backend(self):
        """Test topk with auto backend selection"""
        query_scores = np.array([5.0, 2.0, 8.0, 1.0, 6.0], dtype=np.float32)
        
        scores, indices = topk(query_scores, k=3, backend="auto", sorted=True)
        
        # Should return top 3: 8.0, 6.0, 5.0
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(indices), 3)
        # Check that the top score is 8.0
        self.assertEqual(scores[0], 8.0)


if __name__ == "__main__":
    unittest.main()
