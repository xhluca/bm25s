import os
import unittest
import bm25s


class TestBM25SUtilityFunctions(unittest.TestCase):
    """Test coverage for utility functions in __init__.py"""

    def test_is_list_of_list_of_type_empty_list(self):
        """Test is_list_of_list_of_type with empty list"""
        result = bm25s.is_list_of_list_of_type([])
        self.assertFalse(result)

    def test_is_list_of_list_of_type_not_list(self):
        """Test is_list_of_list_of_type with non-list"""
        result = bm25s.is_list_of_list_of_type("not a list")
        self.assertFalse(result)

    def test_is_list_of_list_of_type_empty_inner_list(self):
        """Test is_list_of_list_of_type with empty inner list"""
        result = bm25s.is_list_of_list_of_type([[]])
        self.assertFalse(result)

    def test_is_list_of_list_of_type_wrong_type(self):
        """Test is_list_of_list_of_type with wrong type"""
        result = bm25s.is_list_of_list_of_type([["string"]], type_=int)
        self.assertFalse(result)

    def test_is_list_of_list_of_type_correct_type(self):
        """Test is_list_of_list_of_type with correct type"""
        result = bm25s.is_list_of_list_of_type([[1, 2, 3]], type_=int)
        self.assertTrue(result)

    def test_is_tuple_of_list_of_tokens_empty_tuple(self):
        """Test _is_tuple_of_list_of_tokens with empty tuple"""
        result = bm25s._is_tuple_of_list_of_tokens(())
        self.assertFalse(result)

    def test_is_tuple_of_list_of_tokens_not_tuple(self):
        """Test _is_tuple_of_list_of_tokens with non-tuple"""
        result = bm25s._is_tuple_of_list_of_tokens([])
        self.assertFalse(result)

    def test_is_tuple_of_list_of_tokens_empty_list(self):
        """Test _is_tuple_of_list_of_tokens with empty list inside"""
        result = bm25s._is_tuple_of_list_of_tokens(([],))
        self.assertFalse(result)

    def test_is_tuple_of_list_of_tokens_not_list_inside(self):
        """Test _is_tuple_of_list_of_tokens with non-list inside"""
        result = bm25s._is_tuple_of_list_of_tokens(("not a list",))
        self.assertFalse(result)

    def test_is_tuple_of_list_of_tokens_not_string(self):
        """Test _is_tuple_of_list_of_tokens with non-string tokens"""
        result = bm25s._is_tuple_of_list_of_tokens(([1, 2, 3],))
        self.assertFalse(result)

    def test_is_tuple_of_list_of_tokens_valid(self):
        """Test _is_tuple_of_list_of_tokens with valid input"""
        result = bm25s._is_tuple_of_list_of_tokens((["hello", "world"],))
        self.assertTrue(result)

    def test_get_unique_tokens(self):
        """Test get_unique_tokens function"""
        corpus_tokens = [["hello", "world"], ["foo", "bar"], ["hello", "foo"]]
        unique = bm25s.get_unique_tokens(corpus_tokens, show_progress=False)
        
        self.assertIsInstance(unique, set)
        self.assertEqual(len(unique), 4)
        self.assertIn("hello", unique)
        self.assertIn("world", unique)
        self.assertIn("foo", unique)
        self.assertIn("bar", unique)

    def test_faketqdm_when_tqdm_disabled(self):
        """Test that fake tqdm works when DISABLE_TQDM is set"""
        # Save original value
        original_tqdm = os.environ.get("DISABLE_TQDM")
        
        try:
            # Set DISABLE_TQDM
            os.environ["DISABLE_TQDM"] = "1"
            
            # Re-import to get the fake tqdm
            import importlib
            importlib.reload(bm25s)
            
            # Test that tqdm just returns the iterable
            test_list = [1, 2, 3]
            # The fake tqdm should just return the iterable as-is
            self.assertIs(bm25s.tqdm(test_list), test_list)
        finally:
            # Restore original value
            if original_tqdm is None:
                os.environ.pop("DISABLE_TQDM", None)
            else:
                os.environ["DISABLE_TQDM"] = original_tqdm
            
            # Reload to restore normal behavior
            importlib.reload(bm25s)


if __name__ == "__main__":
    unittest.main()
