import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from bm25s.utils.beir import download_dataset


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"content-length": str(len(data) if length is None else length)}


class TestDatasetDownload(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.archive = self.root / "archive" / "toy.zip"

    def download(self, responses, **kwargs):
        with patch("urllib.request.urlopen", side_effect=responses):
            return download_dataset(
                "toy", save_dir=self.root, unzip=False, show_progress=False, **kwargs
            )

    def test_failed_download_can_be_retried(self):
        with self.assertRaises(OSError):
            self.download([OSError("connection lost")])
        self.assertFalse(self.archive.exists())
        self.assertEqual(self.download([Response(b"complete")]).read_bytes(), b"complete")

    def test_short_redownload_preserves_cached_archive(self):
        self.download([Response(b"old")])
        with self.assertRaises(OSError):
            self.download([Response(b"short", length=20)], redownload=True)
        self.assertEqual(self.archive.read_bytes(), b"old")

    def test_failed_multipart_download_leaves_no_cached_parts(self):
        missing = HTTPError("test", 404, "missing", {}, None)
        with self.assertRaises(OSError):
            self.download([missing, Response(b"first"), OSError("connection lost")])
        self.assertEqual(list(self.archive.parent.iterdir()), [])
        path = self.download([missing, Response(b"first"), Response(b"last"), missing])
        self.assertEqual(path.read_bytes(), b"firstlast")
