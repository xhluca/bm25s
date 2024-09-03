import os
import shutil
from pathlib import Path
import unittest
import tempfile
import unittest.mock
import Stemmer  # optional: for stemming
import json

import bm25s
from bm25s.utils import json_functions

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
    def test_load_and_save_mmindex(self):
        import bm25s

        try:
            import orjson
        except ImportError:
            self.fail("orjson is not installed")

        file = self.file
        mmindex = bm25s.utils.corpus.find_newline_positions(file)
        bm25s.utils.corpus.save_mmindex(mmindex, file)

        # read the first line
        mmindex = bm25s.utils.corpus.load_mmindex(file)
        
        for i in range(500):
            self.assertEqual(bm25s.utils.corpus.get_line(file, i, mmindex), self.strings[i])
    
    @unittest.mock.patch("bm25s.utils.json_functions.dumps", json_functions.dumps_with_builtin)
    @unittest.mock.patch("bm25s.utils.json_functions.loads", json.loads)
    def test_load_and_save_mmindex_no_orjson(self):
        self.assertEqual(json_functions.dumps_with_builtin, json_functions.dumps)
        self.assertEqual(json_functions.loads, json.loads)
        self.test_load_and_save_mmindex()
    
    def tearDown(self):
        # remove the temp dir with rmtree
        shutil.rmtree(self.tmpdirname)