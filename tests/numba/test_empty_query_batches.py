import tempfile
import unittest

import numpy as np

import bm25s


class TestEmptyQueryBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for method in ('lucene', 'robertson', 'atire', 'bm25l', 'bm25+'):
            model = bm25s.BM25(backend='numba', method=method, auto_compile=False)
            model.index(
                bm25s.tokenize(['cat purrs', 'dog barks', 'fish swims'], show_progress=False),
                show_progress=False,
            )
            cls.models[method] = model

    def assert_batch(self, model, queries):
        result = model.retrieve(queries, k=3, show_progress=False, n_threads=1)
        self.assertEqual(result.scores.shape, (len(queries), 3))
        self.assertEqual(result.scores.dtype, np.dtype(model.dtype))
        self.assertEqual(result.documents.dtype, np.dtype(model.int_dtype))
        for i, query in enumerate(queries):
            self.assertEqual(len(set(result.documents[i].tolist())), 3)
            if query:
                expected = model.retrieve([query], k=3, show_progress=False, n_threads=1)
                np.testing.assert_array_equal(result.documents[i], expected.documents[0])
                np.testing.assert_array_equal(result.scores[i], expected.scores[0])
                self.assertGreater(result.scores[i, 0], 0)
            else:
                np.testing.assert_array_equal(result.scores[i], 0)

    def test_stopword_queries_keep_batch_positions(self):
        for method, model in self.models.items():
            for texts in (
                ['the', 'cat', 'dog'],
                ['the', 'and', 'cat'],
                ['cat', 'the', 'dog'],
                ['cat', 'dog', 'the'],
                ['the', 'and'],
            ):
                tokenized = bm25s.tokenize(texts, show_progress=False)
                strings = bm25s.tokenization.convert_tokenized_to_string_list(tokenized)
                with self.subTest(method=method, texts=texts, format='strings'):
                    self.assert_batch(model, strings)
                for queries in (tokenized, tuple(tokenized)):
                    with self.subTest(method=method, texts=texts, format=type(queries).__name__):
                        result = model.retrieve(queries, k=3, show_progress=False, n_threads=1)
                        expected = model.retrieve(strings, k=3, show_progress=False, n_threads=1)
                        np.testing.assert_array_equal(result.documents, expected.documents)
                        np.testing.assert_array_equal(result.scores, expected.scores)

    def test_leading_empty_id_query(self):
        for method, model in self.models.items():
            with self.subTest(method=method):
                self.assert_batch(model, [[], [model.vocab_dict['cat']], [model.vocab_dict['dog']]])

    def test_save_load_preserves_empty_query_batches(self):
        model = self.models['lucene']
        with tempfile.TemporaryDirectory() as path:
            model.save(path, show_progress=False)
            for mmap in (False, True):
                with self.subTest(mmap=mmap):
                    loaded = bm25s.BM25.load(path, mmap=mmap, auto_compile=False)
                    self.assert_batch(loaded, [[], ['cat'], ['dog']])


class TestDefaultFlowQueryValidation(unittest.TestCase):
    """Default tokenize + index flow, where the empty token is added after the CSC index."""

    @classmethod
    def setUpClass(cls):
        cls.model = bm25s.BM25(backend='numba', auto_compile=False)
        cls.model.index(
            bm25s.tokenize(['cat purrs', 'dog barks', 'fish swims'], show_progress=False),
            show_progress=False,
        )

    def retrieve(self, queries):
        return self.model.retrieve(queries, k=3, show_progress=False, n_threads=1)

    def test_empty_token_fallback_is_outside_default_index(self):
        # Guards the precondition the regression tests below depend on.
        self.assertEqual(self.model.vocab_dict[''], len(self.model.scores['indptr']) - 1)

    def test_out_of_index_ids_raise_with_and_without_empty_rows(self):
        empty_id = self.model.vocab_dict['']
        for queries in (
            [[999]], [[], [999]], [[999], []],
            [[], [empty_id]], [[empty_id], []], [[], [0], [999]],
        ):
            with self.subTest(queries=queries):
                with self.assertRaisesRegex(ValueError, 'outside the index'):
                    self.retrieve(queries)

    def test_out_of_index_error_keeps_numba_thread_count(self):
        from numba import config, get_num_threads

        if config.NUMBA_NUM_THREADS < 2:
            self.skipTest('needs at least two numba threads')
        before = get_num_threads()
        with self.assertRaisesRegex(ValueError, 'outside the index'):
            self.model.retrieve([[], [999]], k=3, show_progress=False, n_threads=2)
        self.assertEqual(get_num_threads(), before)

    def test_in_index_rows_keep_batch_positions(self):
        cat, dog = self.model.vocab_dict['cat'], self.model.vocab_dict['dog']
        for queries in ([[], [cat]], [[cat], []], [[], [dog, cat]], [[], []], [[], ['cat']], [['dog'], [], ['cat']]):
            with self.subTest(queries=queries):
                result = self.retrieve(queries)
                self.assertEqual(result.scores.shape, (len(queries), 3))
                for row, query in enumerate(queries):
                    if query:
                        expected = self.retrieve([query])
                        np.testing.assert_array_equal(result.scores[row], expected.scores[0])
                        np.testing.assert_array_equal(result.documents[row], expected.documents[0])
                    else:
                        np.testing.assert_array_equal(result.scores[row], 0)

    def test_mixed_and_non_list_batches_raise_documented_error(self):
        for queries in (
            [[], ['cat'], [0]],
            [[0], ['cat']],
            [[], [0, 'cat']],
            [[], ['cat', 0]],
            [['cat'], [], [0]],
            [[], np.array([], dtype=int)],
            [np.array([], dtype=int), [0]],
            [np.array([0]), np.array([2])],
            [],
        ):
            with self.subTest(queries=queries):
                with self.assertRaisesRegex(ValueError, 'list of list of tokens'):
                    self.retrieve(queries)


if __name__ == '__main__':
    unittest.main()
