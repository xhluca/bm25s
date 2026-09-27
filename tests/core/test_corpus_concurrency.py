import json
import mmap
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from bm25s.utils.corpus import JsonlCorpus, find_newline_positions, get_line


class InterleavedMmap:
    """Wrap a real mmap to force two cursor-based reads to interleave."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.barrier = Barrier(2)

    def seek(self, position):
        self.mapping.seek(position)
        self.barrier.wait(timeout=5)

    def __getitem__(self, index):
        return self.mapping[index]

    def __getattr__(self, name):
        return getattr(self.mapping, name)


class TestCorpusConcurrency(unittest.TestCase):
    def test_interleaved_reads(self):
        # Both old seek calls finish before either readline can start.
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corpus.jsonl")
            documents = [{"id": i, "text": "document " + str(i)} for i in range(3)]
            with open(path, "wb") as handle:
                handle.write(
                    "".join(json.dumps(doc) + "\n" for doc in documents).encode()
                )
            corpus = JsonlCorpus(path, show_progress=False, verbosity=0)
            try:
                corpus.mmap_obj = InterleavedMmap(corpus.mmap_obj)
                with ThreadPoolExecutor(2) as pool:
                    results = list(pool.map(corpus.__getitem__, [0, 1]))
                self.assertEqual(results, documents[:2])
            finally:
                corpus.close()

    def test_shared_corpus_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corpus.jsonl")
            documents = [{"id": i, "text": "document " + str(i)} for i in range(100)]
            with open(path, "wb") as handle:
                handle.write("\n".join(json.dumps(doc) for doc in documents).encode())
            corpus = JsonlCorpus(path, show_progress=False, verbosity=0)
            try:
                indices = list(range(100)) * 20 + [-1, -100]
                with ThreadPoolExecutor(8) as pool:
                    results = list(pool.map(corpus.__getitem__, indices))
                self.assertEqual(results, [documents[i] for i in indices])
                corpus.close()
                corpus.load()
                self.assertEqual(corpus[:], documents)
            finally:
                corpus.close()

    def test_line_boundaries(self):
        for newline in (b"\n", b"\r\n"):
            for trailing in (False, True):
                with self.subTest(newline=newline, trailing=trailing):
                    with tempfile.TemporaryDirectory() as directory:
                        path = os.path.join(directory, "corpus.jsonl")
                        lines = [
                            json.dumps({"text": text}, ensure_ascii=False).encode(
                                "utf-8"
                            )
                            for text in ("caf\u00e9", "\u4e2d\u6587", "last")
                        ]
                        data = newline.join(lines) + (newline if trailing else b"")
                        with open(path, "wb") as handle:
                            handle.write(data)
                        positions = find_newline_positions(path, show_progress=False)
                        expected = [
                            line + (newline if i < 2 or trailing else b"")
                            for i, line in enumerate(lines)
                        ]
                        with open(path, "rb") as handle:
                            with mmap.mmap(
                                handle.fileno(), 0, access=mmap.ACCESS_READ
                            ) as mapping:
                                for i in (0, 1, 2, -1, -3):
                                    result = get_line(
                                        path,
                                        i,
                                        positions,
                                        file_obj=handle,
                                        mmap_obj=mapping,
                                    )
                                    self.assertEqual(
                                        result.encode("utf-8"), expected[i]
                                    )
                        self.assertEqual(
                            get_line(path, -1, positions).encode("utf-8"), expected[-1]
                        )
