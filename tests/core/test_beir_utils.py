import os
import tempfile
import unittest

from bm25s.utils.beir import load_qrels


class TestLoadQrels(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        qrels_dir = os.path.join(self.tmpdir.name, "toy", "qrels")
        os.makedirs(qrels_dir)
        with open(os.path.join(qrels_dir, "test.tsv"), "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            f.write("q1\td1\t1\n")
            f.write("q1\td2\t1\n")
            f.write("q2\td3\t2\n")
            f.write("q1\td4\t2\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_return_dict_keeps_all_relevant_docs_per_query(self):
        qrels = load_qrels("toy", save_dir=self.tmpdir.name, show_progress=False)

        self.assertEqual(
            qrels,
            {
                "q1": {"d1": 1, "d2": 1, "d4": 2},
                "q2": {"d3": 2},
            },
        )

    def test_return_list_is_unchanged(self):
        qrels = load_qrels("toy", save_dir=self.tmpdir.name, show_progress=False, return_dict=False)

        self.assertEqual(
            qrels,
            [("q1", "d1", 1), ("q1", "d2", 1), ("q2", "d3", 2), ("q1", "d4", 2)],
        )


if __name__ == "__main__":
    unittest.main()
