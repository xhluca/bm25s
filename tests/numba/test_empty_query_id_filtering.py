import unittest

import numpy as np

import bm25s


class TestEmptyQueryIDFiltering(unittest.TestCase):
    def make_model(self, method='bm25+', dtype='float32', int_dtype='int32'):
        # Reserve a real empty column before indexing, independent of PR202.
        corpus = bm25s.tokenization.Tokenized(
            ids=[[0, 1], [2, 3], [4, 5]],
            vocab={'cat': 0, 'purrs': 1, 'dog': 2, 'barks': 3,
                   'fish': 4, 'swims': 5, '': 6},
        )
        model = bm25s.BM25(
            backend='numba', method=method, dtype=dtype, int_dtype=int_dtype,
            auto_compile=False,
        )
        model.index(corpus, show_progress=False)
        return model

    def test_unknown_ids_are_filtered_independently_of_empty_row_position(self):
        for method in ('lucene', 'robertson', 'atire', 'bm25l', 'bm25+'):
            for dtype, int_dtype in (('float32', 'int32'), ('float64', 'int64')):
                model = self.make_model(method, dtype, int_dtype)
                # Negative IDs can silently select nonoccurrence scores; positive
                # out-of-range IDs are also checked with NUMBA_BOUNDSCHECK=1.
                for unknown in ([-2], [18], [-2, 18]):
                    for empty_position in (0, 1, 2):
                        with self.subTest(method=method, dtype=dtype,
                                          unknown=unknown, position=empty_position):
                            queries = [[0, *unknown, 0], [2, *unknown]]
                            queries.insert(empty_position, [])
                            original = [q.copy() for q in queries]
                            result = model.retrieve(
                                queries, k=3, show_progress=False, n_threads=1,
                            )
                            self.assertEqual(queries, original)
                            self.assertEqual(result.scores.shape, (3, 3))
                            self.assertEqual(result.scores.dtype, np.dtype(dtype))
                            self.assertEqual(result.documents.dtype, np.dtype(int_dtype))
                            for row, query in enumerate(queries):
                                filtered = [i for i in query if i in model.unique_token_ids_set]
                                expected = model.get_scores_from_ids(filtered)
                                scores_by_id = result.scores[row][np.argsort(result.documents[row])]
                                np.testing.assert_allclose(scores_by_id, expected, rtol=1e-6)
                                self.assertEqual(len(set(result.documents[row].tolist())), 3)

    def test_unknown_only_rows_keep_existing_empty_token_fallback(self):
        model = self.make_model()
        for query in ([-2], [18], [-2, 18]):
            expected = model.retrieve([query], k=3, show_progress=False, n_threads=1)
            for queries, row in (([[], query], 1), ([query, []], 0)):
                with self.subTest(query=query, row=row):
                    result = model.retrieve(queries, k=3, show_progress=False, n_threads=1)
                    np.testing.assert_array_equal(result.documents[row], expected.documents[0])
                    np.testing.assert_array_equal(result.scores[row], expected.scores[0])
                    np.testing.assert_array_equal(result.scores[1 - row], 0)

    def test_leading_empty_ids_require_loaded_vocabulary(self):
        for missing in (False, True):
            model = self.make_model()
            if missing:
                del model.unique_token_ids_set
            else:
                model.unique_token_ids_set = None
            for queries in ([[], [0]], [[0], []]):
                with self.subTest(missing=missing, queries=queries):
                    with self.assertRaisesRegex(ValueError, 'unique_token_ids_set'):
                        model.retrieve(queries, k=3, show_progress=False, n_threads=1)

    def test_unknown_only_ids_without_empty_token_keep_existing_error(self):
        model = self.make_model()
        del model.vocab_dict['']
        model.unique_token_ids_set = set(model.vocab_dict.values())
        for queries in ([[], [18]], [[18], []]):
            with self.subTest(queries=queries):
                with self.assertRaisesRegex(ValueError, 'does not contain any tokens'):
                    model.retrieve(queries, k=3, show_progress=False, n_threads=1)


if __name__ == '__main__':
    unittest.main()
