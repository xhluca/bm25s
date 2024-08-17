import os
import shutil
from pathlib import Path
import unittest
import tempfile
import bm25s
import Stemmer  # optional: for stemming

class TestBM25SLoadingSaving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create your corpus here
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
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

    @classmethod
    def tearDownClass(cls):
        # remove the temp dir with rmtree
        shutil.rmtree(cls.tmpdirname)