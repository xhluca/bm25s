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


if __name__ == '__main__':
    unittest.main()
