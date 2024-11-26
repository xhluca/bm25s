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

        cls.text =["Thanks for your great work!"] # this works fine
        cls.text = ['שלום חברים'] # this crashes!
        
        # create a vocabulary
        tokens = [ t.split() for t in cls.text ]
        unique_tokens = set([item for sublist in tokens for item in sublist])
        vocab_token2id = {token: i for i, token in enumerate(unique_tokens)}

        # create a tokenized corpus
        token_ids = [ [vocab_token2id[token] for token in text_tokens if token in vocab_token2id] for text_tokens in tokens ]        
        corpus_tokens = bm25s.tokenization.Tokenized(ids=token_ids, vocab=vocab_token2id)

        # create a retriever
        cls.retriever = bm25s.BM25()
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
        self.retriever.save(self.tmpdirname, corpus=self.text) 
        self.retriever.load(self.tmpdirname, load_corpus=True)
    
    @classmethod
    def tearDownClass(cls):
        # remove the temp dir with rmtree
        shutil.rmtree(cls.tmpdirname)