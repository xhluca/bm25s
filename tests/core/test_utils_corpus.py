import os
import shutil
from pathlib import Path
import unittest
import tempfile
import unittest.mock
import Stemmer  # optional: for stemming
import json

class TestUtilsCorpus(unittest.TestCase):
    def setUp(self):
        # let's test the functions
        # random jsonl file
        self.tmpdirname = tempfile.mkdtemp()
        file = os.path.join(self.tmpdirname, "file.jsonl")
        self.file = file
        # content is random uuids
        import uuid

        self.strings = []

        with open(file, "w") as f:
            for i in range(500):
                s = str(json.dumps({"uuid": str(uuid.uuid4())})) + "\n"
                self.strings.append(s)
                f.write(s)

    # hide orjson from importable
    def test_load_and_save_mmindex_with_orjson(self):
        import bm25s

        file = self.file

        mmindex = bm25s.utils.corpus.find_newline_positions(file)
        bm25s.utils.corpus.save_mmindex(mmindex, file)

        # read the first line
        mmindex = bm25s.utils.corpus.load_mmindex(file)
        
        for i in range(500):
            self.assertEqual(bm25s.utils.corpus.get_line(file, i, mmindex), self.strings[i])

    @unittest.mock.patch.dict("sys.modules", {"orjson": None})
    def test_load_and_save_mmindex_without_orjson(self):
        # assert fail that orjson is not installed
        with self.assertRaises(Exception):
            import orjson
        
        import bm25s

        file = self.file

        mmindex = bm25s.utils.corpus.find_newline_positions(file)
        bm25s.utils.corpus.save_mmindex(mmindex, file)

        # read the first line
        mmindex = bm25s.utils.corpus.load_mmindex(file)
        
        for i in range(500):
            self.assertEqual(bm25s.utils.corpus.get_line(file, i, mmindex), self.strings[i])
    
    def tearDown(self):
        # remove the temp dir with rmtree
        shutil.rmtree(self.tmpdirname)