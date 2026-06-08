import unittest

import numpy as np


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
class TestCuPySparseBatchRetrieve(unittest.TestCase):
    def test_sparse_batch_topk_accumulates_duplicate_terms(self):
        from bm25s.cupy.retrieve_utils import _retrieve_sparse_batch_cupy

        # CSC score matrix with shape docs x terms.
        # term 0: doc 0 -> 1, doc 2 -> 2
        # term 1: doc 1 -> 5, doc 2 -> 1
        # term 2: doc 0 -> 4, doc 3 -> 3
        scores = {
            "data": np.asarray([1, 2, 5, 1, 4, 3], dtype=np.float32),
            "indices": np.asarray([0, 2, 1, 2, 0, 3], dtype=np.int32),
            "indptr": np.asarray([0, 2, 4, 6], dtype=np.int32),
            "num_docs": 4,
        }
        query_tokens_ids = [
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 0, 2], dtype=np.int32),
        ]

        docs, returned_scores = _retrieve_sparse_batch_cupy(
            query_tokens_ids=query_tokens_ids,
            scores=scores,
            k=2,
            dtype=np.dtype("float32"),
            int_dtype=np.dtype("int32"),
            max_postings=100,
        )

        expected_docs = np.asarray([[1, 2], [0, 2]], dtype=np.int32)
        expected_scores = np.asarray([[5, 3], [6, 4]], dtype=np.float32)
        np.testing.assert_array_equal(docs, expected_docs)
        np.testing.assert_allclose(returned_scores, expected_scores)


if __name__ == "__main__":
    unittest.main()
