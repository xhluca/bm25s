import os
import subprocess
import sys
import textwrap
import unittest


class TestNumbaThreadLimit(unittest.TestCase):
    def test_all_threads_respects_configured_limit(self):
        code = textwrap.dedent(
            """
            from unittest.mock import patch

            import numpy as np
            from numba import config, get_num_threads, set_num_threads
            import bm25s
            from bm25s.high_level import BM25Search

            model = bm25s.BM25(backend="numba", auto_compile=False)
            model.index([["cat"], ["dog"]], show_progress=False)
            queries = [["cat"], ["dog"]]
            expected = model.retrieve(
                queries, k=1, n_threads=1, show_progress=False
            )
            # A temporary thread mask must also be restored after retrieval.
            set_num_threads(1)
            before = get_num_threads()
            # Also exercises the failure on single-CPU test runners.
            with patch("os.cpu_count", return_value=config.NUMBA_NUM_THREADS + 1):
                for n_threads in (-1, 0, 1, config.NUMBA_NUM_THREADS):
                    result = model.retrieve(
                        queries, k=1, n_threads=n_threads, show_progress=False
                    )
                    np.testing.assert_array_equal(result.documents, expected.documents)
                    np.testing.assert_array_equal(result.scores, expected.scores)
                    assert get_num_threads() == before

                search = BM25Search(["cat purrs", "dog barks"])
                search.show_progress = False
                assert search.search(["cat"], k=1, n_jobs=-1) == search.search(
                    ["cat"], k=1, n_jobs=1
                )
                assert get_num_threads() == before

                # NumPy still resolves -1 independently of Numba's pool.
                model.backend = "numpy"
                result = model.retrieve(
                    queries, k=1, n_threads=-1, backend_selection="numpy",
                    show_progress=False,
                )
                np.testing.assert_array_equal(result.documents, expected.documents)
                np.testing.assert_array_equal(result.scores, expected.scores)

            # Explicit positive counts are not silently clamped.
            model.backend = "numba"
            try:
                model.retrieve(
                    queries, k=1, n_threads=config.NUMBA_NUM_THREADS + 1,
                    show_progress=False,
                )
            except ValueError as error:
                assert "number of threads" in str(error)
            else:
                raise AssertionError("an explicit count above the limit must fail")
            assert get_num_threads() == before

            # A retrieval failure must not change the caller's thread mask.
            try:
                model.retrieve(
                    queries, k=-1, n_threads=-1, show_progress=False
                )
            except ValueError:
                pass
            else:
                raise AssertionError("negative k must fail")
            assert get_num_threads() == before, "thread mask changed after failure"
            """
        )
        for limit in (1, 2):
            with self.subTest(limit=limit):
                env = dict(os.environ, NUMBA_NUM_THREADS=str(limit))
                env.pop("NUMBA_DISABLE_JIT", None)
                result = subprocess.run(
                    [sys.executable, "-B", "-c", code],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
