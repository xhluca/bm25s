import os
import shutil
from pathlib import Path
import unittest
import tempfile
import Stemmer  # optional: for stemming
import unittest.mock
from .test_save_load import TestBM25SLoadingSaving

@unittest.mock.patch.dict("sys.modules", {"orjson": None})
class TestBM25SLoadingSavingWithoutOrjson(TestBM25SLoadingSaving):
    @unittest.mock.patch.dict("sys.modules", {"orjson": None})
    def setUp(self):
        # verify that orjson is not installed
        with self.assertRaises(ImportError, msg="orjson should not be installed"):
            import orjson
    
    @unittest.mock.patch.dict("sys.modules", {"orjson": None})
    def test_a_save(self):
        super().test_a_save()
    
    @unittest.mock.patch.dict("sys.modules", {"orjson": None})
    def test_b_load(self):
        super().test_b_load()
        