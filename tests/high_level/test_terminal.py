import unittest
import tempfile
import shutil
import os
import json
import csv
import sys
from io import StringIO
from argparse import Namespace

from bm25s.terminal import index_command, search_command, create_parser, main


class TestTerminalCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.data_dir = self.tmpdirname
        self.index_dir = os.path.join(self.tmpdirname, "test_index")
        
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

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_create_parser(self):
        """Test that the parser is created correctly."""
        parser = create_parser()
        self.assertIsNotNone(parser)
        
        # Test index subcommand parsing
        args = parser.parse_args(["index", "test.csv", "-o", "output_dir"])
        self.assertEqual(args.command, "index")
        self.assertEqual(args.file, "test.csv")
        self.assertEqual(args.output, "output_dir")
        
        # Test search subcommand parsing
        args = parser.parse_args(["search", "--index=myindex", "query text"])
        self.assertEqual(args.command, "search")
        self.assertEqual(args.index, "myindex")
        self.assertEqual(args.query, "query text")

    def test_index_txt_file(self):
        """Test indexing a TXT file."""
        args = Namespace(
            file=self.txt_path,
            output=self.index_dir,
            column=None
        )
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        # Check that index was created
        self.assertTrue(os.path.exists(self.index_dir))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "params.index.json")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "corpus.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "vocab.tokenizer.json")))
        
        # Check output messages
        self.assertIn("Loaded 3 documents", output)
        self.assertIn("Index saved successfully", output)

    def test_index_csv_file(self):
        """Test indexing a CSV file with column specification."""
        args = Namespace(
            file=self.csv_path,
            output=self.index_dir,
            column="text"
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        self.assertTrue(os.path.exists(self.index_dir))
        self.assertIn("Loaded 3 documents", output)

    def test_index_jsonl_file(self):
        """Test indexing a JSONL file."""
        args = Namespace(
            file=self.jsonl_path,
            output=self.index_dir,
            column="text"
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        self.assertTrue(os.path.exists(self.index_dir))
        self.assertIn("Loaded 3 documents", output)

    def test_index_json_file(self):
        """Test indexing a JSON file."""
        args = Namespace(
            file=self.json_path,
            output=self.index_dir,
            column="text"
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        self.assertTrue(os.path.exists(self.index_dir))
        self.assertIn("Loaded 3 documents", output)

    def test_index_default_output(self):
        """Test that default output directory is created correctly."""
        args = Namespace(
            file=self.txt_path,
            output=None,  # Default output
            column=None
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        old_cwd = os.getcwd()
        os.chdir(self.tmpdirname)
        
        try:
            index_command(args)
        finally:
            sys.stdout = old_stdout
            os.chdir(old_cwd)
        
        # Default output should be corpus_index
        default_index = os.path.join(self.tmpdirname, "corpus_index")
        self.assertTrue(os.path.exists(default_index))
        
        # Clean up
        shutil.rmtree(default_index)

    def test_index_file_not_found(self):
        """Test error handling for missing input file."""
        args = Namespace(
            file="/nonexistent/path/to/file.txt",
            output=self.index_dir,
            column=None
        )
        
        with self.assertRaises(SystemExit) as context:
            index_command(args)
        
        self.assertEqual(context.exception.code, 1)

    def test_search_command(self):
        """Test searching an index."""
        # First, create an index
        index_args = Namespace(
            file=self.txt_path,
            output=self.index_dir,
            column=None
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(index_args)
        finally:
            sys.stdout = old_stdout
        
        # Now search
        search_args = Namespace(
            index=self.index_dir,
            query="hello world",
            top_k=10
        )
        
        sys.stdout = StringIO()
        
        try:
            search_command(search_args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        # Check output
        self.assertIn("Search results for:", output)
        self.assertIn("hello world", output)
        self.assertIn("score:", output)

    def test_search_with_top_k(self):
        """Test searching with custom top-k value."""
        # First, create an index
        index_args = Namespace(
            file=self.txt_path,
            output=self.index_dir,
            column=None
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(index_args)
        finally:
            sys.stdout = old_stdout
        
        # Search with k=2
        search_args = Namespace(
            index=self.index_dir,
            query="test",
            top_k=2
        )
        
        sys.stdout = StringIO()
        
        try:
            search_command(search_args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        self.assertIn("Showing top 2 of 3 documents", output)

    def test_search_index_not_found(self):
        """Test error handling for missing index directory."""
        args = Namespace(
            index="/nonexistent/path/to/index",
            query="test query",
            top_k=10
        )
        
        with self.assertRaises(SystemExit) as context:
            search_command(args)
        
        self.assertEqual(context.exception.code, 1)

    def test_full_workflow_via_main(self):
        """Test the full workflow using the main() function."""
        # Index
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            main(["index", self.txt_path, "-o", self.index_dir])
        finally:
            sys.stdout = old_stdout
        
        self.assertTrue(os.path.exists(self.index_dir))
        
        # Search
        sys.stdout = StringIO()
        
        try:
            main(["search", "--index", self.index_dir, "fast"])
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        self.assertIn("fast", output)
        self.assertIn("bm25s is fast", output)

    def test_main_no_command(self):
        """Test main with no command prints help."""
        with self.assertRaises(SystemExit) as context:
            main([])
        
        self.assertEqual(context.exception.code, 1)

    def test_search_relevance_ranking(self):
        """Test that search returns results in relevance order."""
        # Create an index
        index_args = Namespace(
            file=self.txt_path,
            output=self.index_dir,
            column=None
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(index_args)
        finally:
            sys.stdout = old_stdout
        
        # Search for "fast" - should rank "bm25s is fast" first
        search_args = Namespace(
            index=self.index_dir,
            query="fast",
            top_k=3
        )
        
        sys.stdout = StringIO()
        
        try:
            search_command(search_args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        # The first result should contain "fast"
        lines = output.split("\n")
        first_result_idx = None
        for i, line in enumerate(lines):
            if "[1]" in line:
                first_result_idx = i
                break
        
        self.assertIsNotNone(first_result_idx)
        # Check that "fast" appears in the document after [1]
        first_doc_line = lines[first_result_idx + 1] if first_result_idx + 1 < len(lines) else ""
        self.assertIn("fast", first_doc_line)


class TestTerminalCLIEdgeCases(unittest.TestCase):
    """Test edge cases for the terminal CLI."""
    
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.index_dir = os.path.join(self.tmpdirname, "test_index")

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_empty_file(self):
        """Test handling of empty input file."""
        empty_file = os.path.join(self.tmpdirname, "empty.txt")
        with open(empty_file, "w", encoding="utf-8") as f:
            f.write("")
        
        args = Namespace(
            file=empty_file,
            output=self.index_dir,
            column=None
        )
        
        with self.assertRaises(SystemExit) as context:
            index_command(args)
        
        self.assertEqual(context.exception.code, 1)

    def test_long_document_truncation(self):
        """Test that long documents are truncated in search output."""
        # Create a file with a very long document
        long_doc = "word " * 100  # 500 characters
        long_file = os.path.join(self.tmpdirname, "long.txt")
        with open(long_file, "w", encoding="utf-8") as f:
            f.write(long_doc + "\n")
            f.write("short doc\n")
        
        args = Namespace(
            file=long_file,
            output=self.index_dir,
            column=None
        )
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            index_command(args)
        finally:
            sys.stdout = old_stdout
        
        # Search and check truncation
        search_args = Namespace(
            index=self.index_dir,
            query="word",
            top_k=2
        )
        
        sys.stdout = StringIO()
        
        try:
            search_command(search_args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        # Should have "..." for truncated content
        self.assertIn("...", output)


if __name__ == "__main__":
    unittest.main()

