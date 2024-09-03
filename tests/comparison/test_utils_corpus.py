import json
import logging
import unittest

from tqdm import tqdm
import beir.util

from bm25s.utils.corpus import JsonlCorpus
from bm25s.utils.beir import BASE_URL

class TestTopKSingleQuery(unittest.TestCase):
    def test_utils_corpus(self):
        save_dir = "datasets"
        dataset = "scifact"
        data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)

        corpus_path = f"{data_path}/corpus.jsonl"

        nq = JsonlCorpus(corpus_path)

        # get all ids

        corpus_ids = [doc["_id"] for doc in tqdm(nq)]

        # alternatively, try opening the file and read the _ids as we go
        corpus_ids_2 = []
        with open(corpus_path, "r") as f:
            for line in f:
                doc = json.loads(line)
                corpus_ids_2.append(doc["_id"])

        self.assertListEqual(corpus_ids, corpus_ids_2)

        # check if jsonl corpus can be closed
        assert nq.file_obj is not None, "JsonlCorpus file_obj is None, expected file object"
        assert nq.mmap_obj is not None, "JsonlCorpus mmap_obj is None, expected mmap object"

        # now, we can close
        nq.close()

        assert nq.file_obj is None, "JsonlCorpus file_obj is not None, expected None"
        assert nq.mmap_obj is None, "JsonlCorpus mmap_obj is not None, expected None"

        # check if jsonl corpus can be loaded
        nq.load()

        assert nq.file_obj is not None, "JsonlCorpus file_obj is None, expected file object"
        assert nq.mmap_obj is not None, "JsonlCorpus mmap_obj is None, expected mmap object"

        corpus_ids = [doc["_id"] for doc in tqdm(nq)]
        self.assertListEqual(corpus_ids, corpus_ids_2)

