import os
import shutil
from pathlib import Path
import unittest
import tempfile

import numpy as np
import bm25s
import Stemmer  # optional: for stemming

class TestNumbaBackendRetrieve(unittest.TestCase):
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
        retriever = bm25s.BM25(method='bm25+', backend="numba", corpus=corpus)
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

    def test_b_retrieve_with_numba(self):
        # load the retriever from temp dir
        retriever = bm25s.BM25.load(
            self.tmpdirname,
            data_name="data.index.csc.npy",
            indices_name="indices.index.csc.npy",
            indptr_name="indptr.index.csc.npy",
            vocab_name="vocab.json",
            nnoc_name="nonoccurrence_array.npy",
            params_name="params.json",
            load_corpus=True,
        )

        self.assertTrue(retriever.backend == "numba", "The backend should be 'numba'")

        reloaded_corpus_text = [c["text"] for c in retriever.corpus]
        self.assertTrue(reloaded_corpus_text == self.corpus, "The corpus should be the same as the original corpus")

        # now, let's retrieve the top-k results for a query
        query = ["my cat loves to purr", "a fish likes swimming"]
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        
        # retrieve the top-k results
        top_k = 2
        retrieved = retriever.retrieve(query_tokens, k=top_k, return_as="tuple")
        retrieved_docs = retriever.retrieve(query_tokens, k=top_k, return_as="documents")

        # now, let's retrieve the top-k results for a query using numpy
        retriever.backend = "numpy"
        retrieved_np = retriever.retrieve(query_tokens, k=top_k, return_as="tuple")
        retrieved_docs_np = retriever.retrieve(query_tokens, k=top_k, return_as="documents")
        # assert that the results are the same
        self.assertTrue(np.all(retrieved.scores == retrieved_np.scores), "The retrieved scores should be the same")
        self.assertTrue(np.all(retrieved.documents == retrieved_np.documents), "The retrieved documents should be the same")
        self.assertTrue(np.all(retrieved_docs == retrieved_docs_np), "The results should be the same")
        
    # finally, check when it's loaded with mmap
    def test_c_mmap_retrieve_with_numba(self):
        # load the retriever from temp dir
        retriever = bm25s.BM25.load(
            self.tmpdirname,
            data_name="data.index.csc.npy",
            indices_name="indices.index.csc.npy",
            indptr_name="indptr.index.csc.npy",
            vocab_name="vocab.json",
            nnoc_name="nonoccurrence_array.npy",
            params_name="params.json",
            load_corpus=True,
            mmap=True
        )

        self.assertTrue(retriever.backend == "numba", "The backend should be 'numba'")

        reloaded_corpus_text = [c["text"] for c in retriever.corpus]
        self.assertTrue(reloaded_corpus_text == self.corpus, "The corpus should be the same as the original corpus")

        # now, let's retrieve the top-k results for a query
        query = ["my cat loves to purr", "a fish likes swimming"]
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        
        # retrieve the top-k results
        top_k = 2
        retrieved = retriever.retrieve(query_tokens, k=top_k, return_as="tuple")
        retrieved_docs = retriever.retrieve(query_tokens, k=top_k, return_as="documents")

        # now, let's retrieve the top-k results for a query using numpy
        retriever.backend = "numpy"
        retrieved_np = retriever.retrieve(query_tokens, k=top_k, return_as="tuple")
        retrieved_docs_np = retriever.retrieve(query_tokens, k=top_k, return_as="documents")
        # assert that the results are the same
        self.assertTrue(np.all(retrieved.scores == retrieved_np.scores), "The retrieved scores should be the same")
        self.assertTrue(np.all(retrieved.documents == retrieved_np.documents), "The retrieved documents should be the same")
        self.assertTrue(np.all(retrieved_docs == retrieved_docs_np), "The results should be the same")


    @classmethod
    def tearDownClass(cls):
        # remove the temp dir with rmtree
        shutil.rmtree(cls.tmpdirname)