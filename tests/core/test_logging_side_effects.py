import logging
import tempfile
import unittest
from pathlib import Path

import bm25s
from bm25s.utils.corpus import JsonlCorpus


class TestLibraryLeavesLoggingAlone(unittest.TestCase):
    """Using the library must not configure the application's root logger.

    The module-level ``logging.info()`` implicitly calls ``logging.basicConfig()`` when the root
    logger has no handlers, which adds a stderr handler and makes a later ``logging.basicConfig()``
    in the application a silent no-op.
    """

    def setUp(self):
        root = logging.getLogger()
        saved_handlers, saved_level = list(root.handlers), root.level

        def restore():
            root.handlers[:] = saved_handlers
            root.setLevel(saved_level)

        self.addCleanup(restore)
        root.handlers.clear()

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_loading_a_memory_mapped_corpus_does_not_configure_the_root_logger(self):
        corpus = ["a cat purrs", "a dog barks", "a bird sings"]
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
        retriever.save(self.directory, corpus=corpus)

        loaded = bm25s.BM25.load(self.directory, load_corpus=True, mmap=True)
        self.addCleanup(loaded.corpus.close)

        self.assertEqual(logging.getLogger().handlers, [])

    def test_jsonl_corpus_does_not_configure_the_root_logger(self):
        path = self.directory / "corpus.jsonl"
        path.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")

        corpus = JsonlCorpus(path, show_progress=False, verbosity=1)
        self.addCleanup(corpus.close)
        corpus.close()

        self.assertEqual(logging.getLogger().handlers, [])

    def test_jsonl_corpus_messages_go_through_the_bm25s_logger(self):
        path = self.directory / "corpus.jsonl"
        path.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")

        with self.assertLogs("bm25s", level="INFO") as logs:
            corpus = JsonlCorpus(path, show_progress=False, verbosity=1)
            self.addCleanup(corpus.close)
            corpus.close()

        messages = [record.getMessage() for record in logs.records]
        self.assertIn("Creating index file for jsonl corpus", messages)
        self.assertIn("Opened file and mmap objects", messages)
        self.assertIn("Closed file and mmap objects", messages)
