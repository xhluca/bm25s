import os
import shutil
import tempfile
import unittest

import numpy as np
import Stemmer

import bm25s


def _cupy_runtime_available():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        cp.asnumpy(cp.asarray([1], dtype=cp.float32))
    except Exception as exc:
        return False, f"CuPy runtime is not available: {exc}"
    return True, ""


CUPY_AVAILABLE, CUPY_SKIP_REASON = _cupy_runtime_available()


@unittest.skipUnless(CUPY_AVAILABLE, CUPY_SKIP_REASON)
class TestCuPyBackendRetrieve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ]

        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        retriever = bm25s.BM25(method="bm25+", backend="cupy", corpus=corpus)
        retriever.index(corpus_tokens)

        cls.retriever = retriever
        cls.corpus = corpus
        cls.corpus_tokens = corpus_tokens
        cls.stemmer = stemmer
        cls.tmpdirname = tempfile.mkdtemp()

    def test_a_save(self):
        self.retriever.save(
            self.tmpdirname,
            data_name="data.index.csc.npy",
            indices_name="indices.index.csc.npy",
            indptr_name="indptr.index.csc.npy",
            vocab_name="vocab.json",
            nnoc_name="nonoccurrence_array.npy",
            params_name="params.json",
        )

        fnames = [
            "data.index.csc.npy",
            "indices.index.csc.npy",
            "indptr.index.csc.npy",
            "vocab.json",
            "nonoccurrence_array.npy",
            "params.json",
        ]

        for fname in fnames:
            error_msg = (
                f"File {fname} not found even though it should be saved by .save()"
            )
            path_exists = os.path.exists(os.path.join(self.tmpdirname, fname))
            self.assertTrue(path_exists, error_msg)

    def test_b_retrieve_with_cupy(self):
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

        self.assertTrue(retriever.backend == "cupy", "The backend should be 'cupy'")

        reloaded_corpus_text = [c["text"] for c in retriever.corpus]
        self.assertTrue(
            reloaded_corpus_text == self.corpus,
            "The corpus should be the same as the original corpus",
        )

        query = ["my cat loves to purr", "a fish likes swimming"]
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)

        top_k = 2
        retrieved = retriever.retrieve(query_tokens, k=top_k, return_as="tuple")
        retrieved_docs = retriever.retrieve(query_tokens, k=top_k, return_as="documents")

        retriever.backend = "numpy"
        retrieved_np = retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="tuple",
            backend_selection="numpy",
        )
        retrieved_docs_np = retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="documents",
            backend_selection="numpy",
        )
        retrieved_cupy_selection = retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="tuple",
            backend_selection="cupy",
        )

        np.testing.assert_allclose(retrieved.scores, retrieved_np.scores)
        np.testing.assert_array_equal(retrieved.documents, retrieved_np.documents)
        np.testing.assert_array_equal(retrieved_docs, retrieved_docs_np)
        np.testing.assert_allclose(
            retrieved_cupy_selection.scores, retrieved_np.scores
        )
        np.testing.assert_array_equal(
            retrieved_cupy_selection.documents, retrieved_np.documents
        )

    def test_c_mmap_retrieve_with_cupy(self):
        retriever = bm25s.BM25.load(
            self.tmpdirname,
            data_name="data.index.csc.npy",
            indices_name="indices.index.csc.npy",
            indptr_name="indptr.index.csc.npy",
            vocab_name="vocab.json",
            nnoc_name="nonoccurrence_array.npy",
            params_name="params.json",
            load_corpus=True,
            mmap=True,
        )

        self.assertTrue(retriever.backend == "cupy", "The backend should be 'cupy'")

        reloaded_corpus_text = [c["text"] for c in retriever.corpus]
        self.assertTrue(
            reloaded_corpus_text == self.corpus,
            "The corpus should be the same as the original corpus",
        )

        query = ["my cat loves to purr", "a fish likes swimming"]
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)

        top_k = 2
        retrieved = retriever.retrieve(query_tokens, k=top_k, return_as="tuple")
        retrieved_docs = retriever.retrieve(query_tokens, k=top_k, return_as="documents")

        retriever.backend = "numpy"
        retrieved_np = retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="tuple",
            backend_selection="numpy",
        )
        retrieved_docs_np = retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="documents",
            backend_selection="numpy",
        )
        retrieved_cupy_selection = retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="tuple",
            backend_selection="cupy",
        )

        np.testing.assert_allclose(retrieved.scores, retrieved_np.scores)
        np.testing.assert_array_equal(retrieved.documents, retrieved_np.documents)
        np.testing.assert_array_equal(retrieved_docs, retrieved_docs_np)
        np.testing.assert_allclose(
            retrieved_cupy_selection.scores, retrieved_np.scores
        )
        np.testing.assert_array_equal(
            retrieved_cupy_selection.documents, retrieved_np.documents
        )

    def test_d_retrieve_with_weight_mask(self):
        for dt in [np.float32, np.int32, np.bool_]:
            weight_mask = np.array([1, 1, 0, 1], dtype=dt)
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

            self.assertTrue(
                retriever.backend == "cupy", "The backend should be 'cupy'"
            )

            query = ["my cat loves to purr", "a fish likes swimming"]
            query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)

            top_k = 2
            retrieved = retriever.retrieve(
                query_tokens, k=top_k, return_as="tuple", weight_mask=weight_mask
            )

            retriever.backend = "numpy"
            retrieved_np = retriever.retrieve(
                query_tokens,
                k=top_k,
                return_as="tuple",
                weight_mask=weight_mask,
                backend_selection="numpy",
            )

            np.testing.assert_allclose(retrieved.scores, retrieved_np.scores)
            np.testing.assert_array_equal(retrieved.documents, retrieved_np.documents)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)


if __name__ == "__main__":
    unittest.main()
