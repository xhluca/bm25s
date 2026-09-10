import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestNogil(unittest.TestCase):
    def test_environment_setting(self):
        # Each setting needs a fresh import because decorators run at import time.
        script = """
import sys
import numpy as np
import bm25s
from bm25s.numba import retrieve_utils, selection

expected = sys.argv[1] == "1"
retriever = bm25s.BM25(auto_compile=False)
retriever.compile()
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
    assert function.targetoptions["nogil"] == expected, function.py_func.__name__

retriever.index([["cat"], ["dog"], ["fish"]], show_progress=False)
for backend in ("numpy", "numba"):
    retriever.backend = backend
    result = retriever.retrieve([["dog"]], k=1, n_threads=1, show_progress=False)
    np.testing.assert_array_equal(result.documents, [[1]])
    assert result.scores[0, 0] > 0
"""
        for value in (None, "0", "1"):
            with self.subTest(BM25S_NOGIL=value):
                env = os.environ.copy()
                env.pop("BM25S_NOGIL", None)
                env.pop("NUMBA_DISABLE_JIT", None)
                if value is not None:
                    env["BM25S_NOGIL"] = value
                result = subprocess.run(
                    [sys.executable, "-c", script, value or "0"],
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
