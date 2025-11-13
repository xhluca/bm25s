import unittest
from bm25s.utils import json_functions


class TestJsonFunctions(unittest.TestCase):
    """Test coverage for json_functions utilities"""

    def test_dumps_with_builtin(self):
        """Test dumps_with_builtin function"""
        data = {"key": "value", "number": 42}
        result = json_functions.dumps_with_builtin(data)
        self.assertIsInstance(result, str)
        self.assertIn("key", result)
        self.assertIn("value", result)

    def test_dumps_with_builtin_ensure_ascii(self):
        """Test dumps_with_builtin with ensure_ascii parameter"""
        data = {"text": "hello world"}
        result = json_functions.dumps_with_builtin(data, ensure_ascii=True)
        self.assertIsInstance(result, str)

    def test_dumps_function(self):
        """Test dumps function (uses orjson if available, else builtin)"""
        data = {"test": "data", "num": 123}
        result = json_functions.dumps(data)
        self.assertIsInstance(result, str)
        
        # Should be able to load it back
        loaded = json_functions.loads(result)
        self.assertEqual(loaded["test"], "data")
        self.assertEqual(loaded["num"], 123)

    def test_dumps_with_unicode(self):
        """Test dumps with unicode characters"""
        data = {"text": "héllo wörld"}
        
        # With ensure_ascii=False
        result = json_functions.dumps(data, ensure_ascii=False)
        self.assertIsInstance(result, str)
        
        # Should be able to load it back
        loaded = json_functions.loads(result)
        self.assertEqual(loaded["text"], "héllo wörld")

    def test_loads_function(self):
        """Test loads function"""
        json_str = '{"key": "value", "number": 42}'
        result = json_functions.loads(json_str)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)

    def test_dumps_with_orjson_ensure_ascii_true(self):
        """Test dumps_with_orjson with ensure_ascii=True"""
        # Only run if orjson is available
        if json_functions.ORJSON_AVAILABLE:
            data = {"text": "héllo"}
            result = json_functions.dumps_with_orjson(data, ensure_ascii=True)
            self.assertIsInstance(result, str)
            # Should have escaped non-ASCII characters
            self.assertIn("\\", result)

    def test_dumps_without_orjson(self):
        """Test that builtin json is used when orjson is not available"""
        # Test the path when orjson is not available by checking function assignment
        if not json_functions.ORJSON_AVAILABLE:
            # If orjson is not available, dumps should use builtin
            data = {"test": "value"}
            result = json_functions.dumps(data)
            self.assertIsInstance(result, str)
            
            # loads should also use builtin json
            loaded = json_functions.loads(result)
            self.assertEqual(loaded, data)


if __name__ == "__main__":
    unittest.main()
