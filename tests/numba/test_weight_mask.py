import unittest

import numpy as np

import bm25s


@unittest.skipUnless(bm25s.NUMBA_AVAILABLE, "Numba is not installed")
class TestWeightMaskBackendParity(unittest.TestCase):
    def test_weighted_scores_match_across_backends(self):
        for method in ("bm25l", "bm25+"):
            retriever = bm25s.BM25(method=method, delta=1, backend="numpy")
            retriever.index([["apple"], ["banana"], ["banana"]], show_progress=False)
            for mask in (
                np.array([1, 0, 1], dtype=np.bool_),
                np.array([0.25, 1, 0], dtype=np.float32),
            ):
                for queries in (
                    [["apple"]],
                    [["apple"], ["apple", "apple", "banana"]],
                ):
                    expected = np.array(
                        [retriever.get_scores(query) * mask for query in queries]
                    )
                    for backend, n_threads in (
                        ("numpy", 0),
                        ("numba", 1),
                        ("numba", 2),
                    ):
                        with self.subTest(
                            method=method,
                            mask=mask,
                            queries=queries,
                            backend=backend,
                            n_threads=n_threads,
                        ):
                            retriever.backend = backend
                            result = retriever.retrieve(
                                queries,
                                k=3,
                                weight_mask=mask,
                                n_threads=n_threads,
                                backend_selection=backend,
                                show_progress=False,
                            )
                            # Compare by document ID so tied scores need no tie order.
                            scores_by_id = np.take_along_axis(
                                result.scores,
                                np.argsort(result.documents, axis=1),
                                axis=1,
                            )
                            np.testing.assert_allclose(
                                scores_by_id, expected, rtol=1e-6
                            )

    def test_fractional_mask_keeps_top_result_identical_across_backends(self):
        for method in ("bm25l", "bm25+"):
            retriever = bm25s.BM25(method=method, delta=1, backend="numpy")
            retriever.index([["apple"], ["banana"], ["banana"]], show_progress=False)
            mask = np.array([0.25, 1, 0], dtype=np.float32)
            for backend in ("numpy", "numba"):
                with self.subTest(method=method, backend=backend):
                    retriever.backend = backend
                    result = retriever.retrieve(
                        [["apple"]],
                        k=1,
                        weight_mask=mask,
                        backend_selection=backend,
                        n_threads=1,
                        show_progress=False,
                    )
                    np.testing.assert_array_equal(result.documents, [[1]])
