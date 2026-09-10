import unittest
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

import bm25s
from bm25s.numba import retrieve_utils, selection


class TestNogil(unittest.TestCase):
    def test_selection_and_retrieval_nogil_parameter(self):
        scores = np.array([1.0, 3.0, 2.0])
        for nogil in (False, True):
            with self.subTest(nogil=nogil):
                for factory in (selection._get_top_k, retrieve_utils._get_retriever):
                    function = factory(nogil)
                    self.assertEqual(function.targetoptions.get("nogil", False), nogil)
                    self.assertIs(function, factory(nogil))
                values, indices = selection.topk(scores, k=2, nogil=nogil)
                np.testing.assert_array_equal(values, [3.0, 2.0])
                np.testing.assert_array_equal(indices, [1, 2])
        self.assertFalse(selection._get_top_k().targetoptions.get("nogil", False))
        self.assertFalse(retrieve_utils._get_retriever().targetoptions.get("nogil", False))

    def test_compile_nogil_parameter(self):
        retriever = bm25s.BM25(auto_compile=False)
        for nogil in (False, True, False):
            with self.subTest(nogil=nogil):
                retriever.compile(nogil=nogil)
                self.assertEqual(
                    retriever._compute_relevance_from_scores.targetoptions["nogil"], nogil
                )
                self.assertEqual(retriever._np_csc.targetoptions["nogil"], nogil)
        retriever.compile()
        self.assertFalse(retriever._compute_relevance_from_scores.targetoptions["nogil"])
        self.assertFalse(retriever._np_csc.targetoptions["nogil"])

    def test_retrieval_nogil_results(self):
        retriever = bm25s.BM25(backend="numba", auto_compile=False)
        retriever.index([["cat"], ["dog"], ["fish"]], show_progress=False)
        kwargs = dict(k=1, n_threads=1, show_progress=False)
        expected = retriever.retrieve([["dog"]], **kwargs)
        for nogil in (False, True):
            with self.subTest(nogil=nogil):
                result = retriever.retrieve([["dog"]], nogil=nogil, **kwargs)
                np.testing.assert_array_equal(result.documents, expected.documents)
                np.testing.assert_allclose(result.scores, expected.scores)
                self.assertEqual(result.documents[0, 0], 1)

    def test_concurrent_numba_retrieval(self):
        self.check_concurrent_retrieval("numba")

    def test_concurrent_compiled_scorer_retrieval(self):
        self.check_concurrent_retrieval("numpy")

    def check_concurrent_retrieval(self, backend):
        corpus = [["cat", "purr"], ["dog", "play"], ["fish", "swim"]]
        retriever = bm25s.BM25(backend=backend, auto_compile=False)
        retriever.activate_numba_scorer(nogil=True)
        retriever.index(corpus, show_progress=False)
        queries = [["cat"], ["dog"], ["fish"]]

        def retrieve(query):
            return retriever.retrieve([query], k=1, n_threads=1, show_progress=False, nogil=True)

        # Compile before starting threads and establish the serial baseline.
        expected = [retrieve(query) for query in queries]
        if backend == "numba" and numba.threading_layer() == "workqueue":
            self.skipTest("Numba workqueue does not support concurrent parallel calls")
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(retrieve, queries * 4))

        for i, result in enumerate(results):
            baseline = expected[i % len(queries)]
            np.testing.assert_array_equal(result.documents, baseline.documents)
            np.testing.assert_allclose(result.scores, baseline.scores)
            self.assertEqual(result.documents[0, 0], i % len(queries))
