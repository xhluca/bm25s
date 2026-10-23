import unittest

import numpy as np

import bm25s


class TestWeightMask(unittest.TestCase):
    def make_retriever(self, method):
        retriever = bm25s.BM25(method=method, delta=1, backend="numpy")
        retriever.index([["apple"], ["banana"], ["banana"]], show_progress=False)
        return retriever

    def test_zero_mask_zeroes_complete_scores(self):
        for method in ("lucene", "robertson", "atire", "bm25l", "bm25+"):
            for dtype in (np.bool_, np.int32, np.float32):
                with self.subTest(method=method, dtype=dtype):
                    retriever = self.make_retriever(method)
                    mask = np.zeros(3, dtype=dtype)
                    scores = retriever.get_scores(["apple"], weight_mask=mask)
                    np.testing.assert_array_equal(scores, np.zeros(3))

    def test_weights_scale_complete_scores_without_mutating_inputs_or_index(self):
        for method in ("lucene", "robertson", "atire", "bm25l", "bm25+"):
            for weights, dtype in (
                ([1, 0, 1], np.bool_),
                ([2, 0, 1], np.int32),
                ([0.25, 1, 0], np.float32),
                ([0.25, 1, 0], np.float64),
            ):
                with self.subTest(method=method, dtype=dtype):
                    retriever = self.make_retriever(method)
                    query = ["apple", "apple", "banana"]
                    unweighted = retriever.get_scores(query)
                    mask = np.array(weights, dtype=dtype)
                    original_mask = mask.copy()
                    expected = unweighted * mask

                    np.testing.assert_allclose(
                        retriever.get_scores(query, weight_mask=mask),
                        expected,
                        rtol=1e-6,
                    )
                    np.testing.assert_allclose(
                        retriever.get_scores_from_ids(
                            retriever.get_tokens_ids(query), weight_mask=mask
                        ),
                        expected,
                        rtol=1e-6,
                    )
                    np.testing.assert_array_equal(mask, original_mask)
                    np.testing.assert_array_equal(
                        retriever.get_scores(query), unweighted
                    )

    def test_retrieve_selects_top_k_after_weighting_complete_scores(self):
        for method in ("bm25l", "bm25+"):
            with self.subTest(method=method):
                retriever = self.make_retriever(method)
                mask = np.array([0.25, 1, 0], dtype=np.float32)
                expected_scores = retriever.get_scores(["apple"]) * mask
                for k in (1, 3):
                    result = retriever.retrieve(
                        [["apple"]],
                        k=k,
                        weight_mask=mask,
                        backend_selection="numpy",
                        show_progress=False,
                    )
                    expected_ids = np.array([[1, 0, 2]])[:, :k]
                    np.testing.assert_array_equal(result.documents, expected_ids)
                    np.testing.assert_allclose(
                        result.scores, expected_scores[expected_ids], rtol=1e-6
                    )
