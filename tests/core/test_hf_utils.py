import os
import tempfile
import shutil
import unittest
from bm25s.hf import is_dir_empty, can_save_locally


class TestHFUtilities(unittest.TestCase):
    """Test coverage for HuggingFace utility functions"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_is_dir_empty_nonexistent(self):
        """Test is_dir_empty with non-existent directory"""
        non_existent = os.path.join(self.tmpdir, "does_not_exist")
        self.assertTrue(is_dir_empty(non_existent))

    def test_is_dir_empty_empty_dir(self):
        """Test is_dir_empty with empty directory"""
        empty_dir = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty_dir)
        self.assertTrue(is_dir_empty(empty_dir))

    def test_is_dir_empty_non_empty_dir(self):
        """Test is_dir_empty with non-empty directory"""
        non_empty_dir = os.path.join(self.tmpdir, "non_empty")
        os.makedirs(non_empty_dir)
        
        # Add a file
        with open(os.path.join(non_empty_dir, "file.txt"), "w") as f:
            f.write("test")
        
        self.assertFalse(is_dir_empty(non_empty_dir))

    def test_can_save_locally_none_dir(self):
        """Test can_save_locally with None directory"""
        self.assertFalse(can_save_locally(None, overwrite_local=False))
        self.assertFalse(can_save_locally(None, overwrite_local=True))

    def test_can_save_locally_empty_dir(self):
        """Test can_save_locally with empty directory"""
        empty_dir = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty_dir)
        
        self.assertTrue(can_save_locally(empty_dir, overwrite_local=False))
        self.assertTrue(can_save_locally(empty_dir, overwrite_local=True))

    def test_can_save_locally_non_empty_no_overwrite(self):
        """Test can_save_locally with non-empty directory and no overwrite"""
        non_empty_dir = os.path.join(self.tmpdir, "non_empty")
        os.makedirs(non_empty_dir)
        
        # Add a file
        with open(os.path.join(non_empty_dir, "file.txt"), "w") as f:
            f.write("test")
        
        # Should not be able to save without overwrite
        result = can_save_locally(non_empty_dir, overwrite_local=False)
        # The function returns None when it can't save, not False
        self.assertIsNone(result)

    def test_can_save_locally_non_empty_with_overwrite(self):
        """Test can_save_locally with non-empty directory and overwrite"""
        non_empty_dir = os.path.join(self.tmpdir, "non_empty")
        os.makedirs(non_empty_dir)
        
        # Add a file
        with open(os.path.join(non_empty_dir, "file.txt"), "w") as f:
            f.write("test")
        
        self.assertTrue(can_save_locally(non_empty_dir, overwrite_local=True))


if __name__ == "__main__":
    unittest.main()
