import unittest

import numpy as np

from bm25s.cupy.selection import topk as cupy_topk
from bm25s.cupy.selection import _topk_cupy_gpu
from bm25s.cupy.selection import _topk_cupy_sort_gpu
from bm25s.selection import topk as public_topk


def _cupy_runtime_available():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        cp.asnumpy(cp.asarray([1], dtype=cp.float32))
    except Exception as exc:
        return False, f"CuPy runtime is not available: {exc}"
    return True, ""


CUPY_AVAILABLE, CUPY_SKIP_REASON = _cupy_runtime_available()


@unittest.skipUnless(CUPY_AVAILABLE, CUPY_SKIP_REASON)
class TestTopKSingleQueryCuPy(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.k = 5
        self.scores = np.random.uniform(-10, 10, 2000).astype(np.float32)
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

    def test_topk_cupy_sorted(self):
        result_scores, result_indices = cupy_topk(
            self.scores, self.k, backend="cupy", sorted=True
        )
        self.check_results(result_scores, result_indices, sorted=True)

    def test_topk_cupy_unsorted(self):
        result_scores, result_indices = cupy_topk(
            self.scores, self.k, backend="cupy", sorted=False
        )
        self.check_results(result_scores, result_indices, sorted=False)

    def test_public_topk_accepts_cupy_backend(self):
        result_scores, result_indices = public_topk(
            self.scores, self.k, backend="cupy", sorted=True
        )
        self.check_results(result_scores, result_indices, sorted=True)

    def test_topk_cupy_gpu_keeps_results_on_device(self):
        import cupy as cp

        scores_gpu = cp.asarray(self.scores)
        result_scores, result_indices = _topk_cupy_gpu(
            scores_gpu, self.k, sorted=True
        )

        self.assertIsInstance(result_scores, cp.ndarray)
        self.assertIsInstance(result_indices, cp.ndarray)
        self.check_results(
            cp.asnumpy(result_scores),
            cp.asnumpy(result_indices),
            sorted=True,
        )

    def test_topk_cupy_sort_gpu_matches_numpy_sorted_topk(self):
        import cupy as cp

        scores_gpu = cp.asarray(self.scores)
        result_scores, result_indices = _topk_cupy_sort_gpu(
            scores_gpu, self.k, sorted=True
        )

        self.assertIsInstance(result_scores, cp.ndarray)
        self.assertIsInstance(result_indices, cp.ndarray)
        self.check_results(
            cp.asnumpy(result_scores),
            cp.asnumpy(result_indices),
            sorted=True,
        )


if __name__ == "__main__":
    unittest.main()
