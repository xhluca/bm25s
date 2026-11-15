import os
import unittest
from unittest import mock

import numba
import numpy as np

import bm25s


def is_numba_function(fn):
    # numba dispatchers keep the original Python function in `py_func`
    return hasattr(fn, "py_func")


class TestAutoCompile(unittest.TestCase):
    @unittest.skipIf(numba.config.DISABLE_JIT, "numba JIT is disabled in this process")
    def test_auto_compile_activates_numba(self):
        env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_JIT"}
        with mock.patch.dict(os.environ, env, clear=True):
            retriever = bm25s.BM25(backend="numba")

        self.assertTrue(is_numba_function(retriever._np_csc))
        self.assertTrue(is_numba_function(retriever._compute_relevance_from_scores))

        # the compiled functions give the same scores as the default ones
        corpus_tokens = [["cat", "purrs"], ["dog", "barks"], ["cat", "and", "dog"]]
        retriever.index(corpus_tokens, show_progress=False)
        reference = bm25s.BM25(backend="numba", auto_compile=False)
        reference.index(corpus_tokens, show_progress=False)
        np.testing.assert_allclose(
            retriever.get_scores(["cat"]), reference.get_scores(["cat"])
        )

    def test_auto_compile_skipped_when_numba_disable_jit_is_set(self):
        with mock.patch.dict(os.environ, {"NUMBA_DISABLE_JIT": "1"}):
            retriever = bm25s.BM25(backend="numba")

        self.assertFalse(is_numba_function(retriever._np_csc))
        self.assertFalse(is_numba_function(retriever._compute_relevance_from_scores))

    def test_auto_compile_false_keeps_default_functions(self):
        retriever = bm25s.BM25(backend="numba", auto_compile=False)

        self.assertFalse(is_numba_function(retriever._np_csc))
        self.assertFalse(is_numba_function(retriever._compute_relevance_from_scores))


if __name__ == "__main__":
    unittest.main()
