import unittest

import numpy as np

import bm25s
from bm25s.selection import topk


try:
    import cupy  # noqa: F401
except Exception:
    CUPY_INSTALLED = False
else:
    CUPY_INSTALLED = True


@unittest.skipIf(CUPY_INSTALLED, "CuPy is installed")
class TestCuPyMissingDependency(unittest.TestCase):
    def test_topk_cupy_backend_raises_import_error(self):
        query_scores = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)

        with self.assertRaises(ImportError) as context:
            topk(query_scores, k=2, backend="cupy")

        self.assertIn("CuPy", str(context.exception))

    def test_bm25_cupy_backend_raises_import_error(self):
        with self.assertRaises(ImportError) as context:
            bm25s.BM25(backend="cupy")

        self.assertIn("CuPy", str(context.exception))

    def test_retrieve_cupy_selection_raises_import_error(self):
        retriever = bm25s.BM25()
        retriever.index([[0, 1], [1, 2]], show_progress=False)

        with self.assertRaises(ImportError) as context:
            retriever.retrieve(
                [[1]],
                k=1,
                backend_selection="cupy",
                show_progress=False,
            )

        self.assertIn("CuPy", str(context.exception))


if __name__ == "__main__":
    unittest.main()
