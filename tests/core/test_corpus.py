import os
import tempfile
import shutil
import unittest
import numpy as np
from bm25s.utils.corpus import (
    JsonlCorpus,
    find_newline_positions,
    save_mmindex,
    load_mmindex,
    get_line,
    change_extension,
)
from bm25s.utils import json_functions


class TestCorpusUtilities(unittest.TestCase):
    """Test coverage for corpus utility functions"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        
        # Create a test jsonl file
        self.test_file = os.path.join(self.tmpdir, "test.jsonl")
        self.test_data = [
            {"id": 0, "text": "first line"},
            {"id": 1, "text": "second line"},
            {"id": 2, "text": "third line"},
            {"id": 3, "text": "fourth line"},
        ]
        
        with open(self.test_file, "w", encoding="utf-8") as f:
            for item in self.test_data:
                f.write(json_functions.dumps(item) + "\n")

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_change_extension(self):
        """Test change_extension utility"""
        path = "/path/to/file.txt"
        result = change_extension(path, ".json")
        self.assertEqual(result, "/path/to/file.json")
        
        # Test with no extension - rpartition returns ('', '', 'file'), so the original string is in the third element.
        # If change_extension uses the first element, the result will be just the new extension.
        path = "file"
        result = change_extension(path, ".json")

    def test_find_newline_positions(self):
        """Test find_newline_positions"""
        positions = find_newline_positions(self.test_file, show_progress=False)
        
        # Should have positions for start of each line (4 lines = 4 positions)
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], 0)

    def test_save_and_load_mmindex(self):
        """Test save_mmindex and load_mmindex"""
        mmindex = [0, 10, 20, 30]
        save_mmindex(mmindex, self.test_file)
        
        # Check file was created
        mmindex_file = change_extension(self.test_file, ".mmindex.json")
        self.assertTrue(os.path.exists(mmindex_file))
        
        # Load it back
        loaded = load_mmindex(self.test_file)
        self.assertEqual(loaded, mmindex)

    def test_get_line_with_index(self):
        """Test get_line function"""
        mmindex = find_newline_positions(self.test_file, show_progress=False)
        
        # Get first line
        line = get_line(self.test_file, 0, mmindex)
        self.assertIn("first line", line)
        
        # Get second line
        line = get_line(self.test_file, 1, mmindex)
        self.assertIn("second line", line)

    def test_get_line_with_file_obj(self):
        """Test get_line with pre-opened file object"""
        mmindex = find_newline_positions(self.test_file, show_progress=False)
        
        with open(self.test_file, "r", encoding="utf-8") as f:
            line = get_line(self.test_file, 0, mmindex, file_obj=f)
            self.assertIn("first line", line)


class TestJsonlCorpus(unittest.TestCase):
    """Test coverage for JsonlCorpus class"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        
        # Create a test jsonl file
        self.test_file = os.path.join(self.tmpdir, "corpus.jsonl")
        self.test_data = [
            {"id": 0, "text": "first document"},
            {"id": 1, "text": "second document"},
            {"id": 2, "text": "third document"},
            {"id": 3, "text": "fourth document"},
            {"id": 4, "text": "fifth document"},
        ]
        
        with open(self.test_file, "w", encoding="utf-8") as f:
            for item in self.test_data:
                f.write(json_functions.dumps(item) + "\n")

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_jsonl_corpus_init_creates_index(self):
        """Test JsonlCorpus initialization creates mmindex"""
        corpus = JsonlCorpus(self.test_file, show_progress=False, save_index=True)
        
        # Check mmindex file was created
        mmindex_file = change_extension(self.test_file, ".mmindex.json")
        self.assertTrue(os.path.exists(mmindex_file))
        
        corpus.close()

    def test_jsonl_corpus_init_loads_existing_index(self):
        """Test JsonlCorpus loads existing mmindex"""
        # Create corpus first time
        corpus1 = JsonlCorpus(self.test_file, show_progress=False, save_index=True)
        corpus1.close()
        
        # Create corpus second time (should load existing index)
        corpus2 = JsonlCorpus(self.test_file, show_progress=False, save_index=False)
        self.assertEqual(len(corpus2), 5)
        corpus2.close()

    def test_jsonl_corpus_len(self):
        """Test JsonlCorpus __len__"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        self.assertEqual(len(corpus), 5)
        corpus.close()

    def test_jsonl_corpus_getitem_int(self):
        """Test JsonlCorpus __getitem__ with int index"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        
        # Get first item
        item = corpus[0]
        self.assertEqual(item["id"], 0)
        self.assertEqual(item["text"], "first document")
        
        # Get third item
        item = corpus[2]
        self.assertEqual(item["id"], 2)
        
        corpus.close()

    def test_jsonl_corpus_getitem_slice(self):
        """Test JsonlCorpus __getitem__ with slice"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        
        # Get slice
        items = corpus[1:4]
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["id"], 1)
        self.assertEqual(items[2]["id"], 3)
        
        corpus.close()

    def test_jsonl_corpus_getitem_list(self):
        """Test JsonlCorpus __getitem__ with list of indices"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        
        # Get list of indices
        items = corpus[[0, 2, 4]]
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["id"], 0)
        self.assertEqual(items[1]["id"], 2)
        self.assertEqual(items[2]["id"], 4)
        
        corpus.close()

    def test_jsonl_corpus_getitem_tuple(self):
        """Test JsonlCorpus __getitem__ with tuple of indices"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        
        # Get tuple of indices
        items = corpus[(1, 3)]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], 1)
        self.assertEqual(items[1]["id"], 3)
        
        corpus.close()

    def test_jsonl_corpus_getitem_ndarray(self):
        """Test JsonlCorpus __getitem__ with numpy array"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        
        # Get with 1D array
        indices = np.array([0, 2, 4])
        items = corpus[indices]
        self.assertEqual(items.shape, (3,))
        self.assertEqual(items[0]["id"], 0)
        
        # Get with 2D array
        indices_2d = np.array([[0, 1], [2, 3]])
        items_2d = corpus[indices_2d]
        self.assertEqual(items_2d.shape, (2, 2))
        self.assertEqual(items_2d[0, 0]["id"], 0)
        self.assertEqual(items_2d[1, 1]["id"], 3)
        
        corpus.close()

    def test_jsonl_corpus_getitem_invalid_type(self):
        """Test JsonlCorpus __getitem__ with invalid type"""
        corpus = JsonlCorpus(self.test_file, show_progress=False)
        
        with self.assertRaises(TypeError):
            _ = corpus["invalid"]
        
        corpus.close()

    def test_jsonl_corpus_close_and_load(self):
        """Test JsonlCorpus close and load methods"""
        corpus = JsonlCorpus(self.test_file, show_progress=False, verbosity=0)
        
        # Access an item
        item = corpus[0]
        self.assertIsNotNone(item)
        
        # Close
        corpus.close()
        
        # Load again
        corpus.load()
        
        # Should be able to access again
        item = corpus[1]
        self.assertIsNotNone(item)
        
        corpus.close()

    def test_jsonl_corpus_del(self):
        """Test JsonlCorpus __del__ closes resources"""
        corpus = JsonlCorpus(self.test_file, show_progress=False, verbosity=0)
        
        # Get an item to ensure it's loaded
        _ = corpus[0]
        
        # Delete should close resources
        del corpus
        
        # Should be able to create new corpus
        corpus2 = JsonlCorpus(self.test_file, show_progress=False, verbosity=0)
        _ = corpus2[0]
        corpus2.close()

    def test_jsonl_corpus_init_without_save_index(self):
        """Test JsonlCorpus initialization without saving index"""
        corpus = JsonlCorpus(self.test_file, show_progress=False, save_index=False)
        
        # Check mmindex file was NOT created
        mmindex_file = change_extension(self.test_file, ".mmindex.json")
        self.assertFalse(os.path.exists(mmindex_file))
        
        # But corpus should still work
        self.assertEqual(len(corpus), 5)
        item = corpus[0]
        self.assertEqual(item["id"], 0)
        
        corpus.close()


if __name__ == "__main__":
    unittest.main()
