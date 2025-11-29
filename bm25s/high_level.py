"""
The high level BM25 search API. It wraps bm25s into a simple to use search interface,
enabling 1-line indexing and 1-line searching. By default, it will require:
- numba compilation for speed up
- stemming for better search quality
- stopword removal for better search quality
"""

import json
import csv
from pathlib import Path
from . import BM25
from .tokenization import Tokenizer
import Stemmer
from typing import List



class BM25Search:
    def __init__(
        self,
        corpus: List[str],
        language: str = "english",
        bm25_kwargs: dict = None,
        tokenizer_kwargs: dict = None,
        tokenizer_cls: Tokenizer = Tokenizer,
    ):
        if bm25_kwargs is not None and "corpus" in bm25_kwargs:
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
            backend="numba", csc_backend="numpy", auto_compile=False
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
            update_vocab=True,
            return_as="tuple",
        )

        # compile and index
        self.retriever.compile(activate_numba=True, warmup=True)
        
        create_empty_token = True
        # If the corpus is empty or has no tokens, we can't create an empty token
        # as it relies on vocab dict having some content or logic that fails if empty
        if len(tokenized.vocab) == 0:
            create_empty_token = False
            
        self.retriever.index(
            tokenized,
            leave_progress=leave_progress,
            show_progress=show_progress,
            create_empty_token=create_empty_token,
        )

    def search(self, queries: List[str], k: int = 10, n_jobs: int = 1):
        # Ensure k is not larger than the corpus size
        num_docs = len(self.corpus)
        if k > num_docs:
            k = num_docs

        tokenized_queries = self.tokenizer.tokenize(
            queries,
            update_vocab=False,
            show_progress=self.show_progress,
            leave_progress=self.leave_progress,
            return_as="tuple",
        )
        # DEBUG
        print(f"DEBUG: tokenized_queries.ids: {tokenized_queries.ids}")
        
        # Handle empty queries explicitly to avoid issues with Numba backend
        # We filter out queries that result in no tokens
        non_empty_indices = []
        empty_indices = []
        
        # We access internal ids list from Tokenized namedtuple
        # Convert to string list first as retrieve expects that or Tokenized object
        # But checking emptiness is easier on ids
        
        for i, q_ids in enumerate(tokenized_queries.ids):
            if len(q_ids) > 0:
                non_empty_indices.append(i)
            else:
                empty_indices.append(i)
        
        # Prepare results structure
        results = [None] * len(queries)
        
        # Process empty queries immediately
        for i in empty_indices:
            # Empty query results in empty list of documents
            # Or do we want to return empty docs with 0 score? 
            # Standard BM25 usually returns nothing for empty query, or 0 score for all docs.
            # Let's return empty list as top-k results.
            results[i] = []

        if len(non_empty_indices) > 0:
            # Create a new Tokenized object for non-empty queries
            non_empty_ids = [tokenized_queries.ids[i] for i in non_empty_indices]
            non_empty_tokenized = self.tokenizer.to_tokenized_tuple(non_empty_ids)
            
            # Retrieve for non-empty queries
            # note: because we did not pass `corpus` when initializing BM25,
            # the retrieve() will return document ids instead of texts.
            doc_ids, scores = self.retriever.retrieve(
                query_tokens=non_empty_tokenized,
                k=k,
                sorted=True,
                return_as="tuple",
                show_progress=self.show_progress,
                leave_progress=self.leave_progress,
                n_threads=n_jobs,
                chunksize=50,
                backend_selection="auto",
            )
            
            # Map back to original indices
            num_docs = doc_ids.shape[1]
            
            for idx, original_idx in enumerate(non_empty_indices):
                query_results = []
                for di in range(num_docs):
                    doc_id = doc_ids[idx, di]
                    doc_text = self.corpus[doc_id]
                    query_results.append(
                        {
                            "id": int(doc_id),
                            "score": float(scores[idx, di]),
                            "document": doc_text,
                        }
                    )
                results[original_idx] = query_results

        return results


def index(documents, language: str = "english"):
    return BM25Search(corpus=documents, language=language)


def load(path, document_column=None):
    """
    Loads a csv, json, jsonl, or txt file from the given path. For json, we expect a list of dicts.
    Returns a list of strings (corpus) that can be passed to `bm25s.index`.

    Parameters
    ----------
    path : str
        The file path to load the documents from.

    document_column : str, optional
        The column name to use as the document text when loading from csv, or the key when loading from json/jsonl.
        If None, the first column will be used by default.
    
    Returns
    -------
    List[str]
        A list of strings representing the documents.
    """

    path = Path(path)
    documents = []

    if path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]

    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], str):
                    documents = data
                elif len(data) > 0 and isinstance(data[0], dict):
                    if document_column is None:
                        # Use the first key available in the first element
                        document_column = list(data[0].keys())[0]
                    documents = [d[document_column] for d in data]
            else:
                raise ValueError("JSON file must contain a list of strings or dicts.")

    elif path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if document_column is None and not documents:
                    # Infer column from first line
                    document_column = list(d.keys())[0]
                
                if document_column in d:
                    documents.append(d[document_column])
                else:
                    # skip or error? Let's skip if key missing, or error. 
                    # raising error is safer for "load"
                    raise ValueError(f"Key '{document_column}' not found in JSONL line: {line[:50]}...")

    elif path.suffix == ".csv":
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if document_column is None:
                if reader.fieldnames:
                    document_column = reader.fieldnames[0]
                else:
                    return BM25Search(corpus=[])

            for row in reader:
                if document_column in row:
                    documents.append(row[document_column])
                else:
                     raise ValueError(f"Column '{document_column}' not found in CSV.")
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return documents
