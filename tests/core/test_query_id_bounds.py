import unittest

import bm25s
import numpy as np


class TestQueryIdBounds(unittest.TestCase):
    def setUp(self):
        self.model = bm25s.BM25()
        self.model.index([["cat"], ["dog"], ["bird"]], show_progress=False)

    def test_direct_scoring_rejects_negative_ids(self):
        for score in (self.model.get_scores, self.model.get_scores_from_ids):
            for query in ([-1], [-2], [-100], [0, -2]):
                with self.subTest(method=score.__name__, query=query):
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        score(query)

    def test_bounds_are_checked_before_narrowing_ids(self):
        for dtype in (np.int64, np.uint64):
            query = np.array([2**32], dtype=dtype)
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "maximum token ID"):
                    self.model.get_scores_from_ids(query)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.model.get_scores_from_ids(np.array([-2**32], dtype=np.int64))

    def test_existing_upper_bound_is_still_rejected(self):
        first_invalid = len(self.model.scores["indptr"]) - 1
        for score in (self.model.get_scores, self.model.get_scores_from_ids):
            with self.subTest(method=score.__name__):
                with self.assertRaisesRegex(ValueError, "maximum token ID"):
                    score([first_invalid])

    def test_valid_repeated_empty_and_weighted_queries_keep_scores(self):
        cat = self.model.vocab_dict["cat"]
        for dtype in (np.int32, np.int64, np.uint64):
            query = np.array([cat, cat], dtype=dtype)
            original = query.copy()
            with self.subTest(dtype=dtype):
                np.testing.assert_allclose(
                    self.model.get_scores_from_ids(query), [0.7846634, 0, 0], rtol=1e-6
                )
                np.testing.assert_array_equal(query, original)
        np.testing.assert_allclose(self.model.get_scores(["cat"]), [0.3923317, 0, 0])
        np.testing.assert_array_equal(self.model.get_scores_from_ids([]), [0, 0, 0])
        np.testing.assert_array_equal(
            self.model.get_scores([cat], weight_mask=np.array([0, 1, 1])), [0, 0, 0]
        )

    def test_all_scoring_variants_reject_negative_ids(self):
        for method in ("robertson", "lucene", "atire", "bm25l", "bm25+"):
            model = bm25s.BM25(method=method)
            model.index([["cat"], ["dog"], ["bird"]], show_progress=False)
            for score in (model.get_scores, model.get_scores_from_ids):
                with self.subTest(method=method, entry=score.__name__):
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        score([-2])

    @unittest.skipUnless(bm25s.NUMBA_AVAILABLE, "Numba is not installed")
    def test_compiled_scorer_uses_same_validation(self):
        self.model.activate_numba_scorer()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.model.get_scores_from_ids([-2])
        cat = self.model.vocab_dict["cat"]
        np.testing.assert_allclose(self.model.get_scores([cat]), [0.3923317, 0, 0])


if __name__ == "__main__":
    unittest.main()
