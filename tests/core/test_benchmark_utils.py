import builtins
import contextlib
import importlib
import io
import unittest
from unittest import mock

import bm25s.utils.benchmark as benchmark


class TestBenchmarkUtils(unittest.TestCase):
    def test_missing_resource_does_not_write_to_stdout(self):
        original_import = builtins.__import__

        def _import_with_missing_resource(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "resource":
                raise ImportError("resource module not available")
            return original_import(name, globals, locals, fromlist, level)

        mock_logger = mock.Mock()

        try:
            with mock.patch("logging.getLogger", return_value=mock_logger):
                with mock.patch("builtins.__import__", side_effect=_import_with_missing_resource):
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        importlib.reload(benchmark)

            self.assertEqual(stdout.getvalue(), "")
            self.assertIsNone(benchmark.resource)
            self.assertIsNone(benchmark.get_max_memory_usage())
            mock_logger.warning.assert_called_once_with(
                "resource module not available on Windows"
            )
        finally:
            importlib.reload(benchmark)


if __name__ == "__main__":
    unittest.main()
