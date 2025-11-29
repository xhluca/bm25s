import unittest
import tempfile
import shutil
import os
import json
import csv
import bm25s.high_level as bm25_hl

class TestHighLevel(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.data_dir = self.tmpdirname
        
        # Create dummy data
        self.corpus_txt = [
            "hello world",
            "this is a test",
            "bm25s is fast"
        ]
        
        self.corpus_dicts = [
            {"id": 1, "text": "hello world"},
            {"id": 2, "text": "this is a test"},
            {"id": 3, "text": "bm25s is fast"}
        ]
        
        # Write files
        self.txt_path = os.path.join(self.data_dir, "corpus.txt")
        with open(self.txt_path, "w", encoding="utf-8") as f:
            for line in self.corpus_txt:
                f.write(line + "\n")
                
        self.csv_path = os.path.join(self.data_dir, "corpus.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            writer.writeheader()
            for doc in self.corpus_dicts:
                writer.writerow(doc)
                
        self.jsonl_path = os.path.join(self.data_dir, "corpus.jsonl")
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for doc in self.corpus_dicts:
                f.write(json.dumps(doc) + "\n")
                
        self.json_path = os.path.join(self.data_dir, "corpus.json")
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_dicts, f)
            
        self.json_list_path = os.path.join(self.data_dir, "corpus_list.json")
        with open(self.json_list_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_txt, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_load_txt(self):
        corpus = bm25_hl.load(self.txt_path)
        self.assertEqual(len(corpus), 3)
        self.assertEqual(corpus[0], "hello world")
        
        retriever = bm25_hl.index(corpus)
        results = retriever.search(["test"], k=1)
        self.assertEqual(len(results[0]), 1)
        self.assertIn("test", results[0][0]["document"])
        self.assertIsInstance(results[0][0]["score"], float)
        self.assertIsInstance(results[0][0]["id"], int)

    def test_load_csv(self):
        corpus = bm25_hl.load(self.csv_path, document_column="text")
        self.assertEqual(len(corpus), 3)
        self.assertEqual(corpus[0], "hello world")
        
        retriever = bm25_hl.index(corpus)
        results = retriever.search(["fast"], k=1)
        self.assertIn("fast", results[0][0]["document"])

    def test_load_jsonl(self):
        corpus = bm25_hl.load(self.jsonl_path, document_column="text")
        self.assertEqual(len(corpus), 3)
        
        retriever = bm25_hl.index(corpus)
        results = retriever.search(["world"], k=1)
        self.assertIn("world", results[0][0]["document"])

    def test_load_json(self):
        corpus = bm25_hl.load(self.json_path, document_column="text")
        self.assertEqual(len(corpus), 3)
        
        retriever = bm25_hl.index(corpus)
        results = retriever.search(["world"], k=1)
        self.assertIn("world", results[0][0]["document"])
        
    def test_load_json_list(self):
        corpus = bm25_hl.load(self.json_list_path)
        self.assertEqual(len(corpus), 3)
        self.assertEqual(corpus[0], "hello world")

    def test_k_larger_than_corpus(self):
        corpus = bm25_hl.load(self.txt_path)
        retriever = bm25_hl.index(corpus)
        # Corpus has 3 docs, requesting k=10 should not fail
        results = retriever.search(["test"], k=10)
        self.assertEqual(len(results[0]), 3)
        
    def test_empty_query(self):
        corpus = bm25_hl.load(self.txt_path)
        retriever = bm25_hl.index(corpus)
        # Empty query should return empty list
        results = retriever.search([""], k=1)
        self.assertEqual(len(results[0]), 0)
        self.assertEqual(results[0], [])
        
    def test_mixed_empty_query(self):
        corpus = bm25_hl.load(self.txt_path)
        retriever = bm25_hl.index(corpus)
        # One valid, one empty
        results = retriever.search(["test", ""], k=1)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(len(results[1]), 0)

