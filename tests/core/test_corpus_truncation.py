import tempfile
import unittest
from pathlib import Path

import bm25s
from bm25s.utils.corpus import JsonlCorpus, change_extension


class TestCorpusTruncation(unittest.TestCase):
    """A zero-byte file must not load with a nonempty cached line index."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "corpus.jsonl"
        self.path.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")

    def test_constructor_rejects_truncated_file_with_saved_index(self):
        corpus = JsonlCorpus(self.path, show_progress=False, verbosity=0)
        self.addCleanup(corpus.close)
        self.assertEqual(len(corpus), 2)
        self.assertTrue(Path(change_extension(self.path, ".mmindex.json")).is_file())
        corpus.close()
        self.path.write_bytes(b"")

        with self.assertRaises(ValueError):
            reopened = JsonlCorpus(self.path, show_progress=False, verbosity=0)
            self.addCleanup(reopened.close)

    def test_reload_rejects_truncated_file_with_in_memory_index(self):
        corpus = JsonlCorpus(
            self.path, show_progress=False, save_index=False, verbosity=0
        )
        self.addCleanup(corpus.close)
        self.assertEqual(len(corpus), 2)
        corpus.close()
        self.path.write_bytes(b"")

        with self.assertRaises(ValueError):
            corpus.load()
        self.assertIsNone(corpus.file_obj)
        self.assertIsNone(corpus.mmap_obj)

    def test_bm25_load_rejects_truncated_corpus_with_saved_index(self):
        documents = ["alpha beta", "gamma delta"]
        tokens = bm25s.tokenize(documents, show_progress=False)
        model = bm25s.BM25(backend="numpy")
        model.index(tokens, show_progress=False)
        model.save(self.directory, corpus=documents, show_progress=False)

        corpus = JsonlCorpus(self.path, show_progress=False, verbosity=0)
        self.addCleanup(corpus.close)
        self.assertEqual(len(corpus), 2)
        corpus.close()
        self.path.write_bytes(b"")

        with self.assertRaises(ValueError):
            loaded = bm25s.BM25.load(
                self.directory, mmap=True, load_corpus=True, show_progress=False
            )
            self.addCleanup(loaded.corpus.close)


if __name__ == "__main__":
    unittest.main()
