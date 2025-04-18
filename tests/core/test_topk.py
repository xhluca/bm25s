import unittest
import numpy as np

# Assuming JAX_IS_AVAILABLE is a global variable that we need to set for testing
JAX_IS_AVAILABLE = False  # Set to True if you want to test the JAX backend
try:
    import jax
    import jax.numpy as jnp
    JAX_IS_AVAILABLE = True
except ImportError:
    pass

from bm25s.selection import topk


class TestTopKSingleQuery(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.k = 5
        self.scores = np.random.uniform(-10, 10, 2000)
        self.expected_scores = np.sort(self.scores)[-self.k:][::-1]
        self.expected_indices = np.argsort(self.scores)[-self.k:][::-1]

    def check_results(self, result_scores, result_indices, sorted=True):
        if sorted:
            np.testing.assert_allclose(result_scores, self.expected_scores)
            np.testing.assert_array_equal(result_indices, self.expected_indices)
        else:
            self.assertEqual(len(result_scores), self.k)
            self.assertEqual(len(result_indices), self.k)
            self.assertTrue(np.all(np.isin(result_scores, self.expected_scores)))
            self.assertTrue(np.all(np.isin(result_indices, self.expected_indices)))

    def test_topk_numpy_sorted(self):
        result_scores, result_indices = topk(self.scores, self.k, backend="numpy", sorted=True)
        self.check_results(result_scores, result_indices, sorted=True)

    def test_topk_numpy_unsorted(self):
        result_scores, result_indices = topk(self.scores, self.k, backend="numpy", sorted=False)
        self.check_results(result_scores, result_indices, sorted=False)

    @unittest.skipUnless(JAX_IS_AVAILABLE, "JAX is not available")
    def test_topk_jax_sorted(self):
        result_scores, result_indices = topk(jnp.array(self.scores), self.k, backend="jax", sorted=True)
        self.check_results(result_scores, result_indices, sorted=True)

    @unittest.skipUnless(JAX_IS_AVAILABLE, "JAX is not available")
    def test_topk_jax_unsorted(self):
        result_scores, result_indices = topk(jnp.array(self.scores), self.k, backend="jax", sorted=False)
        self.check_results(result_scores, result_indices, sorted=True)

    def test_topk_auto_backend(self):
        result_scores, result_indices = topk(self.scores, self.k, backend="auto", sorted=True)
        self.check_results(result_scores, result_indices, sorted=True)
    
    def test_jax_installed_but_unavailable(self):
        global JAX_IS_AVAILABLE
        original_jax_is_available = JAX_IS_AVAILABLE
        JAX_IS_AVAILABLE = False  # Temporarily pretend JAX is not available
        
        result_scores, result_indices = topk(self.scores, self.k, backend="auto", sorted=True)
        self.check_results(result_scores, result_indices, sorted=True)
        
        JAX_IS_AVAILABLE = original_jax_is_available  # Restore the original value


if __name__ == '__main__':
    unittest.main()
