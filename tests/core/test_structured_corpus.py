import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import bm25s
from bm25s.utils.corpus import JsonlCorpus


class TestStructuredCorpus(unittest.TestCase):
    backend = "numpy"

    def setUp(self):
        self.retriever = bm25s.BM25(backend=self.backend, auto_compile=False)
        self.retriever.index([["cat"], ["dog"], ["bird"]], show_progress=False)
        self.queries = [["cat"], ["dog"]]

    def test_sequence_entries_remain_whole_documents(self):
        corpora = [
            [["cat", 1], ["dog", 2], ["bird", 3]],
            [("cat", 1), ("dog", 2), ("bird", 3)],
            [[1], [2, 3], []],
            [[], [], []],
            [[[1]], [[2]], [[3]]],
            [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],
            [{"text": "cat"}, {"text": "dog"}, {"text": "bird"}],
        ]
        for corpus in corpora:
            with self.subTest(corpus=corpus):
                result = self.retriever.retrieve(
                    self.queries, corpus=corpus, k=1,
                    backend_selection=self.backend, show_progress=False,
                )
                self.assertEqual(result.documents.shape, (2, 1))
                self.assertEqual(result.scores.shape, (2, 1))
                self.assertIs(result.documents[0, 0], corpus[0])
                self.assertIs(result.documents[1, 0], corpus[1])

    def test_scalar_entries_keep_numpy_dtype_inference(self):
        for corpus in (
            [1.5, 2.5, 3.5],
            [10, 20, 30],
            ["cat", "dog", "bird"],
            [np.array(1.5), np.array(2.5), np.array(3.5)],
        ):
            with self.subTest(corpus=corpus):
                documents = self.retriever.retrieve(
                    self.queries, corpus=corpus, k=1, return_as="documents",
                    backend_selection=self.backend, show_progress=False,
                )
                expected = np.array(corpus[:2]).reshape(2, 1)
                self.assertEqual(documents.dtype, expected.dtype)
                np.testing.assert_array_equal(documents, expected)
                if documents.dtype.kind in "fi":
                    self.assertTrue(np.isfinite(documents).all())

    def test_one_dimensional_numpy_corpus_keeps_explicit_dtype(self):
        for dtype in (np.float32, np.int16, "U8"):
            with self.subTest(dtype=dtype):
                corpus = np.array([1, 2, 3], dtype=dtype)
                documents = self.retriever.retrieve(
                    self.queries, corpus=corpus, k=1, return_as="documents",
                    backend_selection=self.backend, show_progress=False,
                )
                self.assertEqual(documents.dtype, corpus.dtype)
                np.testing.assert_array_equal(documents, corpus[:2].reshape(2, 1))

    def test_saved_sequence_entries_work_with_and_without_mmap(self):
        corpus = [["cat", 1], ["dog", 2], ["bird", 3]]
        with tempfile.TemporaryDirectory() as directory:
            self.retriever.save(directory, corpus=corpus, show_progress=False)
            for mmap in (False, True):
                with self.subTest(mmap=mmap):
                    loaded = bm25s.BM25.load(
                        directory, load_corpus=True, mmap=mmap,
                        show_progress=False, verbosity=0,
                    )
                    try:
                        result = loaded.retrieve(
                            self.queries, k=1, backend_selection=self.backend,
                            show_progress=False,
                        )
                        self.assertEqual(result.documents.shape, (2, 1))
                        self.assertEqual(result.documents[0, 0], corpus[0])
                        self.assertEqual(result.documents[1, 0], corpus[1])
                    finally:
                        if mmap:
                            loaded.corpus.close()


class TestJsonlCorpusArrayEntries(unittest.TestCase):
    def test_array_indexing_preserves_entries_and_scalar_dtypes(self):
        for documents in ([1.5, 2.5, 3.5], [[1], [2, 3], []]):
            with self.subTest(documents=documents), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "corpus.jsonl"
                path.write_text(
                    "".join(json.dumps(doc) + "\n" for doc in documents),
                    encoding="utf-8",
                )
                corpus = JsonlCorpus(path, show_progress=False, verbosity=0)
                try:
                    for indices in (
                        np.array([[0, 1], [2, 0]]),
                        np.array(1),
                        np.empty((2, 0), dtype=int),
                    ):
                        result = corpus[indices]
                        self.assertEqual(result.shape, indices.shape)
                        for index, document in zip(indices.flat, result.flat):
                            self.assertEqual(document, documents[index])
                        if isinstance(documents[0], float):
                            self.assertEqual(result.dtype, np.dtype(float))
                            self.assertTrue(np.isfinite(result).all())
                finally:
                    corpus.close()


if __name__ == "__main__":
    unittest.main()
