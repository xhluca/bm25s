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
class TestCuPyCandidateBatchRetrieve(unittest.TestCase):
    def test_candidate_terms_from_query_deduplicates_kept_terms_only(self):
        from bm25s.cupy.retrieve_utils import _candidate_terms_from_query

        indptr = np.asarray([0, 2, 8, 11, 20], dtype=np.int32)
        query_terms = np.asarray([0, 1, 1, 2, 3, 3], dtype=np.int32)

        candidate_terms, dropped_terms = _candidate_terms_from_query(
            query_terms=query_terms,
            indptr_cpu=indptr,
            df_threshold=5,
        )

        np.testing.assert_array_equal(
            candidate_terms, np.asarray([0, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            dropped_terms, np.asarray([1, 1, 3, 3], dtype=np.int32)
        )

    def test_resolve_candidate_certification_batches_kth_transfer(self):
        import cupy as cp

        from bm25s.cupy.retrieve_utils import _resolve_candidate_certification

        topk_scores = [
            cp.asarray([4.0, 3.0, 2.0], dtype=cp.float32),
            cp.asarray([7.0, 6.0, 5.0], dtype=cp.float32),
        ]

        fallback_positions = _resolve_candidate_certification(
            candidate_positions=[10, 20],
            topk_scores=topk_scores,
            outside_bounds=[1.5, 5.0],
        )

        self.assertEqual(fallback_positions, [20])

    def test_build_candidates_atomic_cupy_returns_unique_docs_and_clears_marks(self):
        import cupy as cp

        from bm25s.cupy.retrieve_utils import _build_candidates_atomic_cupy

        # term 0 -> docs [0, 2, 3], term 1 -> docs [2, 4].
        indices_gpu = cp.asarray([0, 2, 3, 2, 4], dtype=cp.int32)
        indptr_cpu = np.asarray([0, 3, 5], dtype=np.int32)
        marks_gpu = cp.zeros(5, dtype=cp.int32)
        count_gpu = cp.zeros(1, dtype=cp.int32)

        candidates = _build_candidates_atomic_cupy(
            candidate_terms=np.asarray([0, 1], dtype=np.int32),
            indptr_cpu=indptr_cpu,
            indices_gpu=indices_gpu,
            marks_gpu=marks_gpu,
            count_gpu=count_gpu,
        )

        np.testing.assert_array_equal(
            np.sort(cp.asnumpy(candidates)),
            np.asarray([0, 2, 3, 4], dtype=np.int32),
        )
        np.testing.assert_array_equal(cp.asnumpy(marks_gpu), np.zeros(5, dtype=np.int32))

    def test_build_candidates_stamp_cupy_reuses_marks_without_clearing(self):
        import cupy as cp

        from bm25s.cupy.retrieve_utils import _build_candidates_stamp_cupy

        indices_gpu = cp.asarray([0, 2, 3, 2, 4], dtype=cp.int32)
        indptr_cpu = np.asarray([0, 3, 5], dtype=np.int32)
        marks_gpu = cp.zeros(5, dtype=cp.int32)
        count_gpu = cp.zeros(1, dtype=cp.int32)

        first = _build_candidates_stamp_cupy(
            candidate_terms=np.asarray([0, 1], dtype=np.int32),
            indptr_cpu=indptr_cpu,
            indices_gpu=indices_gpu,
            marks_gpu=marks_gpu,
            count_gpu=count_gpu,
            stamp=1,
        )
        second = _build_candidates_stamp_cupy(
            candidate_terms=np.asarray([1], dtype=np.int32),
            indptr_cpu=indptr_cpu,
            indices_gpu=indices_gpu,
            marks_gpu=marks_gpu,
            count_gpu=count_gpu,
            stamp=2,
        )

        np.testing.assert_array_equal(
            np.sort(cp.asnumpy(first)),
            np.asarray([0, 2, 3, 4], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.sort(cp.asnumpy(second)),
            np.asarray([2, 4], dtype=np.int32),
        )

    def test_build_candidates_stamp_slots_cupy_keeps_count_on_device(self):
        import cupy as cp

        from bm25s.cupy.retrieve_utils import _build_candidates_stamp_slots_cupy

        indices_gpu = cp.asarray([0, 2, 3, 2, 4], dtype=cp.int32)
        indptr_cpu = np.asarray([0, 3, 5], dtype=np.int32)
        marks_gpu = cp.zeros(5, dtype=cp.int32)
        count_gpu = cp.zeros(1, dtype=cp.int32)

        candidates, total_slots, returned_count = _build_candidates_stamp_slots_cupy(
            candidate_terms=np.asarray([0, 1], dtype=np.int32),
            indptr_cpu=indptr_cpu,
            indices_gpu=indices_gpu,
            marks_gpu=marks_gpu,
            count_gpu=count_gpu,
            stamp=1,
        )

        self.assertEqual(total_slots, 5)
        self.assertIs(returned_count, count_gpu)
        self.assertEqual(int(cp.asnumpy(count_gpu)[0]), 4)
        np.testing.assert_array_equal(
            np.sort(cp.asnumpy(candidates[:4])),
            np.asarray([0, 2, 3, 4], dtype=np.int32),
        )

    def test_unique_sorted_cupy_matches_cupy_unique(self):
        import cupy as cp

        from bm25s.cupy.retrieve_utils import _unique_sorted_cupy

        values = cp.asarray([4, 2, 4, 1, 2, 7, 1], dtype=cp.int32)
        result = _unique_sorted_cupy(values)

        np.testing.assert_array_equal(
            cp.asnumpy(result), np.asarray([1, 2, 4, 7], dtype=np.int32)
        )

    def test_candidate_batch_scores_candidates_with_all_query_terms(self):
        import os

        from bm25s.cupy.retrieve_utils import _retrieve_candidate_batch_cupy

        # CSC score matrix with shape docs x terms.
        # term 0 is the low-df candidate generator.
        # term 1 is a high-df scoring-only term and must still affect ranking.
        scores = {
            "data": np.asarray([5, 4, 0, 10, 1], dtype=np.float32),
            "indices": np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
            "indptr": np.asarray([0, 2, 5], dtype=np.int32),
            "num_docs": 3,
        }
        query_tokens_ids = [np.asarray([0, 1], dtype=np.int32)]
        old = os.environ.get("BM25S_CUPY_CANDIDATE_UNION")
        os.environ["BM25S_CUPY_CANDIDATE_UNION"] = "stamp_slots"
        try:
            docs, returned_scores = _retrieve_candidate_batch_cupy(
                query_tokens_ids=query_tokens_ids,
                scores=scores,
                k=1,
                dtype=np.dtype("float32"),
                int_dtype=np.dtype("int32"),
                df_threshold=2,
            )
        finally:
            if old is None:
                os.environ.pop("BM25S_CUPY_CANDIDATE_UNION", None)
            else:
                os.environ["BM25S_CUPY_CANDIDATE_UNION"] = old

        np.testing.assert_array_equal(docs, np.asarray([[1]], dtype=np.int32))
        np.testing.assert_allclose(
            returned_scores, np.asarray([[14]], dtype=np.float32)
        )

    def test_spmm_candidate_batch_matches_candidate_path_with_dropped_terms(self):
        from bm25s.cupy.retrieve_utils import (
            _retrieve_candidate_batch_cupy,
            _retrieve_spmm_batch_cupy,
        )

        scores = {
            "data": np.asarray([5, 4, 0, 10, 1], dtype=np.float32),
            "indices": np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
            "indptr": np.asarray([0, 2, 5], dtype=np.int32),
            "num_docs": 3,
        }
        query_tokens_ids = [np.asarray([0, 1], dtype=np.int32)]

        expected_docs, expected_scores = _retrieve_candidate_batch_cupy(
            query_tokens_ids=query_tokens_ids,
            scores=scores,
            k=1,
            dtype=np.dtype("float32"),
            int_dtype=np.dtype("int32"),
            df_threshold=2,
        )
        docs, returned_scores = _retrieve_spmm_batch_cupy(
            query_tokens_ids=query_tokens_ids,
            scores=scores,
            k=1,
            dtype=np.dtype("float32"),
            int_dtype=np.dtype("int32"),
            df_threshold=2,
            batch_size=8,
        )

        np.testing.assert_array_equal(docs, expected_docs)
        np.testing.assert_allclose(returned_scores, expected_scores)

    def test_spmm_candidate_batch_scores_duplicate_low_df_terms(self):
        from bm25s.cupy.retrieve_utils import _retrieve_spmm_batch_cupy

        scores = {
            "data": np.asarray([5, 4, 1], dtype=np.float32),
            "indices": np.asarray([0, 1, 2], dtype=np.int32),
            "indptr": np.asarray([0, 2, 3], dtype=np.int32),
            "num_docs": 3,
        }

        docs, returned_scores = _retrieve_spmm_batch_cupy(
            query_tokens_ids=[np.asarray([0, 0, 1], dtype=np.int32)],
            scores=scores,
            k=2,
            dtype=np.dtype("float32"),
            int_dtype=np.dtype("int32"),
            df_threshold=2,
            batch_size=8,
        )

        np.testing.assert_array_equal(docs, np.asarray([[0, 1]], dtype=np.int32))
        np.testing.assert_allclose(returned_scores, np.asarray([[10, 8]], dtype=np.float32))

    def test_topk_from_csr_rows_accepts_precomputed_cpu_indptr(self):
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        from bm25s.cupy.retrieve_utils import _topk_from_csr_rows_cupy

        matrix = cpsp.csr_matrix(
            (
                cp.asarray([1.0, 3.0, 2.0, 4.0], dtype=cp.float32),
                cp.asarray([10, 11, 12, 13], dtype=cp.int32),
                cp.asarray([0, 3, 4], dtype=cp.int32),
            ),
            shape=(2, 20),
        )
        indptr_cpu = cp.asnumpy(matrix.indptr)

        docs_rows, score_rows, fallback_rows = _topk_from_csr_rows_cupy(
            matrix,
            k=2,
            dtype=np.dtype("float32"),
            int_dtype=np.dtype("int32"),
            indptr_cpu=indptr_cpu,
        )

        self.assertEqual(fallback_rows, [1])
        np.testing.assert_array_equal(cp.asnumpy(docs_rows[0]), np.asarray([11, 12], dtype=np.int32))
        np.testing.assert_allclose(cp.asnumpy(score_rows[0]), np.asarray([3.0, 2.0], dtype=np.float32))
        self.assertIsNone(docs_rows[1])
        self.assertIsNone(score_rows[1])


if __name__ == "__main__":
    unittest.main()
