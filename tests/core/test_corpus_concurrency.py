import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from bm25s.utils.corpus import JsonlCorpus, find_newline_positions, get_line


class InterleavedMmap:
    """Let both readers position their reads before either consumes bytes."""

    def __init__(self, mapped):
        self.mapped = mapped
        self.barrier = Barrier(2)

    def seek(self, offset):
        self.mapped.seek(offset)
        self.barrier.wait(timeout=5)

    def readline(self):
        return self.mapped.readline()

    def find(self, value, start):
        self.barrier.wait(timeout=5)
        return self.mapped.find(value, start)

    def __getitem__(self, key):
        return self.mapped[key]

    def __len__(self):
        return len(self.mapped)


class TestCorpusConcurrentReads(unittest.TestCase):
    def test_interleaved_reads_return_requested_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            documents = [{"id": i} for i in range(4)]
            path.write_text(
                "".join(json.dumps(document) + "\n" for document in documents),
                encoding="utf-8",
            )
            corpus = JsonlCorpus(path, show_progress=False, verbosity=0)
            try:
                with patch.object(corpus, "mmap_obj", InterleavedMmap(corpus.mmap_obj)):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        # Nonadjacent indices cannot accidentally pass via consecutive reads.
                        results = list(pool.map(corpus.__getitem__, [0, 2]))
                self.assertEqual(results, [documents[0], documents[2]])
            finally:
                corpus.close()

    def test_reads_preserve_shared_mmap_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text('{"id": 0}\n{"id": 1}\n', encoding="utf-8")
            corpus = JsonlCorpus(path, show_progress=False, verbosity=0)
            try:
                corpus.mmap_obj.seek(3)
                self.assertEqual(corpus[1], {"id": 1})
                self.assertEqual(corpus.mmap_obj.tell(), 3)
            finally:
                corpus.close()

    def test_line_endings_utf8_and_negative_indices(self):
        documents = [{"text": "搜索"}, {"text": "café"}, {"text": "推荐"}]
        for newline in ["\n", "\r\n"]:
            for trailing_newline in [False, True]:
                with self.subTest(newline=newline, trailing_newline=trailing_newline):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "corpus.jsonl"
                        lines = [
                            json.dumps(doc, ensure_ascii=False) for doc in documents
                        ]
                        payload = newline.join(lines) + (
                            newline if trailing_newline else ""
                        )
                        path.write_bytes(payload.encode("utf-8"))
                        offsets = find_newline_positions(path, show_progress=False)
                        for index in range(len(documents)):
                            ending = (
                                newline
                                if index < len(documents) - 1 or trailing_newline
                                else ""
                            )
                            expected = lines[index] + ending
                            self.assertEqual(get_line(path, index, offsets), expected)
                            self.assertEqual(
                                get_line(path, index - len(documents), offsets),
                                expected,
                            )


if __name__ == "__main__":
    unittest.main()
