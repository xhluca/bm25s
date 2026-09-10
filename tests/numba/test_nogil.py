import unittest
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

import bm25s
from bm25s.numba import retrieve_utils, selection


class TestNogil(unittest.TestCase):
    def test_jit_functions_release_gil(self):
        retriever = bm25s.BM25(auto_compile=False)
        retriever.activate_numba_scorer(nogil=True)
        retriever.activate_numba_csc(nogil=True)
        functions = [
            retriever._compute_relevance_from_scores,
            retriever._np_csc,
            retrieve_utils._compute_relevance_from_scores_jit_ready,
            retrieve_utils._retrieve_internal_jitted_parallel,
            selection._numba_unsorted_top_k_legacy,
            selection._numba_sorted_top_k,
            selection.sift_down,
            selection.sift_up,
            selection.heap_push,
            selection.heap_pop,
        ]
        for function in functions:
            with self.subTest(function=function.py_func.__name__):
                self.assertTrue(function.targetoptions.get("nogil", False))

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
            return retriever.retrieve([query], k=1, n_threads=1, show_progress=False)

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
