import os
import shutil
from pathlib import Path
import unittest
import tempfile
import Stemmer  # optional: for stemming
import unittest.mock
import json

import bm25s
from bm25s.utils import json_functions

class TestBM25SLoadingSaving(unittest.TestCase):
    orjson_should_not_be_installed = False
    orjson_should_be_installed = True

    @classmethod
    def setUpClass(cls):
        # check that import orjson fails
        import bm25s

        # Create your corpus here
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
            "שלום חברים, איך אתם היום?",
            "El café está muy caliente",
            "今天的天气真好!",
            "Как дела?",
            "Türkçe öğreniyorum."
        ]

        # optional: create a stemmer
        stemmer = Stemmer.Stemmer("english")

        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25(method='bm25+')
        retriever.index(corpus_tokens)

        # Save the retriever to temp dir
        cls.retriever = retriever
        cls.corpus = corpus
        cls.corpus_tokens = corpus_tokens
        cls.stemmer = stemmer
        cls.tmpdirname = tempfile.mkdtemp()
    
    def setUp(self):
        # verify that orjson is properly installed
        try:
            import orjson
        except ImportError:
            self.fail("orjson should be installed to run this test.")
        
    def test_a_save(self):
        # save the retriever to temp dir
        self.retriever.save(
            self.tmpdirname,
            data_name="data.index.csc.npy",
            indices_name="indices.index.csc.npy",
            indptr_name="indptr.index.csc.npy",
            vocab_name="vocab.json",
            nnoc_name="nonoccurrence_array.npy",
            params_name="params.json",
        )

        # assert that the following files are saved
        fnames = [
            "data.index.csc.npy",
            "indices.index.csc.npy",
            "indptr.index.csc.npy",
            "vocab.json",
            "nonoccurrence_array.npy",
            "params.json",
        ]

        for fname in fnames:
            error_msg = f"File {fname} not found in even though it should be saved by the .save() method"
            path_exists = os.path.exists(os.path.join(self.tmpdirname, fname))
            self.assertTrue(path_exists, error_msg)

    def test_b_load(self):
        # load the retriever from temp dir
        r1 = self.retriever
        r2 = bm25s.BM25.load(
            self.tmpdirname,
            data_name="data.index.csc.npy",
            indices_name="indices.index.csc.npy",
            indptr_name="indptr.index.csc.npy",
            vocab_name="vocab.json",
            nnoc_name="nonoccurrence_array.npy",
            params_name="params.json",
        )

        # for each of data, indices, indptr, vocab, nnoc, params
        # assert that the loaded object is the same as the original object
        # data, indices, indptr are stored in self.scores
        self.assertTrue((r1.scores['data'] == r2.scores['data']).all())
        self.assertTrue((r1.scores['indices'] == r2.scores['indices']).all())
        self.assertTrue((r1.scores['indptr'] == r2.scores['indptr']).all())

        # vocab is stored in self.vocab
        self.assertEqual(r1.vocab_dict, r2.vocab_dict)

        # nnoc is stored in self.nnoc
        self.assertTrue((r1.nonoccurrence_array == r2.nonoccurrence_array).all())
    
    @unittest.mock.patch("bm25s.utils.json_functions.dumps", json_functions.dumps_with_builtin)
    @unittest.mock.patch("bm25s.utils.json_functions.loads", json.loads)
    def test_c_save_no_orjson(self):
        self.assertEqual(json_functions.dumps_with_builtin, json_functions.dumps)
        self.assertEqual(json_functions.loads, json.loads)
        self.test_a_save()
    
    @unittest.mock.patch("bm25s.utils.json_functions.dumps", json_functions.dumps_with_builtin)
    @unittest.mock.patch("bm25s.utils.json_functions.loads", json.loads)
    def test_d_load_no_orjson(self):
        self.assertEqual(json_functions.dumps_with_builtin, json_functions.dumps)
        self.assertEqual(json_functions.loads, json.loads)
        self.test_b_load()

    @classmethod
    def tearDownClass(cls):
        # remove the temp dir with rmtree
        shutil.rmtree(cls.tmpdirname)


class TestBM25SNonASCIILoadingSaving(unittest.TestCase):
    orjson_should_not_be_installed = False
    orjson_should_be_installed = True

    @classmethod
    def setUpClass(cls):
        # check that import orjson fails
        import bm25s
        cls.corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
            "שלום חברים, איך אתם היום?",
            "El café está muy caliente",
            "今天的天气真好!",
            "Как дела?",
            "Türkçe öğreniyorum.",
            'שלום חברים'
        ]
        corpus_tokens = bm25s.tokenize(cls.corpus, stopwords="en")
        cls.retriever = bm25s.BM25(corpus=cls.corpus)
        cls.retriever.index(corpus_tokens)
        cls.tmpdirname = tempfile.mkdtemp()
     
    
    def setUp(self):
        # verify that orjson is properly installed
        try:
            import orjson
        except ImportError:
            self.fail("orjson should be installed to run this test.")

    def test_a_save_and_load(self):
        # both of these fail: UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-6: character maps to <undefined>
        self.retriever.save(self.tmpdirname, corpus=self.corpus) 
        self.retriever.load(self.tmpdirname, load_corpus=True)
    
    @classmethod
    def tearDownClass(cls):
        # remove the temp dir with rmtree
        shutil.rmtree(cls.tmpdirname)


class TestSaveAndReloadWithTokenizer(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.tmpdirname)
    
    def test_save_and_reload_with_tokenizer(self):
        import bm25s
        from bm25s.tokenization import Tokenizer

        corpus = [
            "Welcome to bm25s, a library that implements BM25 in Python, allowing you to rank documents based on a query.",
            "BM25 is a widely used ranking function used for text retrieval tasks, and is a core component of search services like Elasticsearch.",
            "It is designed to be:",
            "Fast: bm25s is implemented in pure Python and leverage Scipy sparse matrices to store eagerly computed scores for all document tokens.",
            "This allows extremely fast scoring at query time, improving performance over popular libraries by orders of magnitude (see benchmarks below).",
            "Simple: bm25s is designed to be easy to use and understand.",
            "You can install it with pip and start using it in minutes.",
            "There is no dependencies on Java or Pytorch - all you need is Scipy and Numpy, and optional lightweight dependencies for stemming.",
            "Below, we compare bm25s with Elasticsearch in terms of speedup over rank-bm25, the most popular Python implementation of BM25.",
            "We measure the throughput in queries per second (QPS) on a few popular datasets from BEIR in a single-threaded setting.",
            "bm25s aims to offer a faster alternative for Python users who need efficient text retrieval.",
            "It leverages modern Python libraries and data structures for performance optimization.",
            "You can find more details in the documentation and example notebooks provided.",
            "Installation and usage guidelines are simple and accessible for developers of all skill levels.",
            "Try bm25s for a scalable and fast text ranking solution in your Python projects."
        ]

        # print(f"We have {len(corpus)} documents in the corpus.")

        tokenizer = Tokenizer(stemmer=None, stopwords=None, splitter=lambda x: x.split())
        corpus_tokens = tokenizer.tokenize(corpus, return_as='tuple')

        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)

        index_path = os.path.join(self.tmpdirname, "bm25s_index_readme")

        retriever.save(index_path)
        tokenizer.save_vocab(save_dir=index_path)

        reloaded_retriever = bm25s.BM25.load(index_path, load_corpus=True)
        reloaded_tokenizer = Tokenizer(stemmer=None, stopwords=None, splitter=lambda x: x.split())
        reloaded_tokenizer.load_vocab(index_path)

        queries = ["widely used text ranking function"]

        query_tokens = reloaded_tokenizer.tokenize(queries, update_vocab=False)
        results, scores = reloaded_retriever.retrieve(query_tokens, k=2)

        doc = results[0,0]
        score = scores[0,0]

        assert doc['id'] == 1
        assert score > 3 and score < 4