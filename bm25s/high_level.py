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


class BM25Search:
    def __init__(
        self, corpus, bm25_kwargs=None, tokenizer_kwargs=None, language="english"
    ):
        self.leave_progress = leave_progress = False
        self.show_progress = show_progress = True

        if language != "english":
            raise NotImplementedError("Currently only English language is supported.")

        stemmer = Stemmer.Stemmer("english")
        bm25_kwargs_default = dict(
            backend="numba", csc_backend="numba", auto_compile=False
        )
        tokenizer_kwargs_default = dict(
            stemmer=stemmer, stopwords="english", lower=True
        )

        if isinstance(bm25_kwargs, dict):
            if "corpus" in bm25_kwargs:
                raise ValueError(
                    "The 'corpus' argument in bm25_kwargs is reserved and cannot be set manually."
                )
            bm25_kwargs_default.update(bm25_kwargs)
        if isinstance(tokenizer_kwargs, dict):
            tokenizer_kwargs_default.update(tokenizer_kwargs)

        # force the corpus to be set
        bm25_kwargs_default["corpus"] = corpus

        self.tokenizer = Tokenizer(**tokenizer_kwargs_default)
        self.retriever = BM25(**bm25_kwargs_default)

        # tokenize the corpus
        token_ids = self.tokenizer.tokenize(
            corpus, leave_progress=leave_progress, show_progress=show_progress
        )

        # compile and index
        self.retriever.compile(activate_numba=True, warmup=True)
        self.retriever.index(
            token_ids,
            leave_progress=leave_progress,
            show_progress=show_progress,
            create_empty_token=True,
        )

    def search(self, queries, k=10, n_jobs=1):
        tokenized_queries = self.tokenizer.tokenize(queries)
        documents, scores = self.retriever.retrieve(
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

        # return as list of dicts with kesy 'document' and 'score'
        results = []
        for doc_list, score_list in zip(documents, scores):
            result = []
            for doc, score in zip(doc_list, score_list):
                if isinstance(doc, dict):
                    doc_copy = doc.copy()
                    doc_copy["score"] = score
                    result.append(doc_copy)
                else:
                    result.append({"document": doc, "score": score})
            results.append(result)
        return results


def index(documents):
    # TODO: implement indexing logic, returns BM25Search instance
    pass


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
