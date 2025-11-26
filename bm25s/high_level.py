"""
The high level BM25 search API. It wraps bm25s into a simple to use search interface,
enabling 1-line indexing and 1-line searching. By default, it will require:
- numba compilation for speed up
- stemming for better search quality
- stopword removal for better search quality
"""

from pathlib import Path
from . import BM25
from .tokenization import Tokenizer
import Stemmer
from typing import List, Tuple, Dict


class BM25Search:
    def __init__(
        self,
        corpus: List[str],
        language: str = "english",
        bm25_kwargs: dict = None,
        tokenizer_kwargs: dict = None,
        tokenizer_cls: Tokenizer = Tokenizer,
    ):
        if "corpus" in bm25_kwargs:
            raise ValueError(
                "The 'corpus' argument in bm25_kwargs is reserved and cannot be set manually."
            )
        if language != "english":
            raise NotImplementedError("Currently only English language is supported.")
        
        self.leave_progress = leave_progress = False
        self.show_progress = show_progress = True
        self.corpus = corpus

        stemmer = Stemmer.Stemmer("english")
        bm25_kwargs_default = dict(
            backend="numba", csc_backend="numba", auto_compile=False
        )
        tokenizer_kwargs_default = dict(
            stemmer=stemmer, stopwords="english", lower=True
        )

        if isinstance(bm25_kwargs, dict):
            bm25_kwargs_default.update(bm25_kwargs)
        elif bm25_kwargs is not None:
            raise ValueError("bm25_kwargs must be a dict or None.")

        if isinstance(tokenizer_kwargs, dict):
            tokenizer_kwargs_default.update(tokenizer_kwargs)
        elif tokenizer_kwargs is not None:
            raise ValueError("tokenizer_kwargs must be a dict or None.")

        if "corpus" in bm25_kwargs_default:
            raise ValueError(
                "The 'corpus' argument in bm25_kwargs is reserved and cannot be set manually."
            )

        # note: we do not pass corpus here, as we will keep it separately. This means the BM25
        # object will return document ids instead of texts when retrieving.
        self.retriever = BM25(**bm25_kwargs_default)
        self.tokenizer: Tokenizer = tokenizer_cls(**tokenizer_kwargs_default)

        # tokenize the corpus
        tokenized = self.tokenizer.tokenize(
            corpus,
            leave_progress=leave_progress,
            show_progress=show_progress,
            update_vocab=False,
            return_as="tuple",
        )

        # compile and index
        self.retriever.compile(activate_numba=True, warmup=True)
        self.retriever.index(
            tokenized,
            leave_progress=leave_progress,
            show_progress=show_progress,
            create_empty_token=True,
        )

    def search(self, queries: List[str], k: int = 10, n_jobs: int = 1):
        tokenized_queries = self.tokenizer.tokenize(
            queries,
            update_vocab=False,
            show_progress=self.show_progress,
            leave_progress=self.leave_progress,
            return_as="tuple",
        )
        # note: because we did not pass `corpus` when initializing BM25,
        # the retrieve() will return document ids instead of texts.
        doc_ids, scores = self.retriever.retrieve(
            query_tokens=tokenized_queries,
            k=k,
            sorted=True,
            return_as="tuple",
            show_progress=self.show_progress,
            leave_progress=self.leave_progress,
            n_threads=n_jobs,
            chunksize=50,
            backend_selection="numpy",
        )

        # return as list of of lists of dicts with keys 'document' and 'score'
        # the outer list is for each query, the inner list is for each of the k documents retrieved for that query
        results = []
        num_queries = doc_ids.shape[0]
        num_docs = doc_ids.shape[1]
        for qi in range(num_queries):
            query_results = []
            for di in range(num_docs):
                doc_id = doc_ids[qi, di]
                doc_text = self.corpus[doc_id]
                query_results.append({"id": doc_id, "score": scores[qi, di], "document": doc_text})
            results.append(query_results)

        return results


def index(documents, language: str = "english"):
    return BM25Search(corpus=documents, language=language)


def load(path, document_column=None):
    """
    Loads a csv, json, jsonl, or txt file from the given path. For json, we expect a list of dicts.

    Parameters
    ----------
    path : str
        The file path to load the documents from.

    document_column : str, optional
        The column name to use as the document text when loading from csv, or the key when loading from json/jsonl.
        If None, the first column will be used by default.
    """

    path = Path(path)
    # TODO: implement loading logic, returns BM25Search instance
    pass
