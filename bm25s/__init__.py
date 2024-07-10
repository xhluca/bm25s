from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial

import os
import logging
from pathlib import Path
from typing import Any, Tuple, Dict, Iterable, List, NamedTuple, Union

import numpy as np

try:
    import ujson as json
except ImportError:
    import json


def _faketqdm(iterable, *args, **kwargs):
    return iterable

if os.environ.get("DISABLE_TQDM", False):
    tqdm = _faketqdm
    # if can't import tqdm, use a fake tqdm
else:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = _faketqdm


from . import selection, utils, stopwords, scoring, tokenization
from .version import __version__
from .tokenization import tokenize
from .scoring import (
    _select_tfc_scorer,
    _select_idf_scorer,
    _build_scores_and_indices_for_matrix,
    _calculate_doc_freqs,
    _build_idf_array,
    _build_nonoccurrence_array,
)

debug_logger = logging.getLogger("bm25s")
debug_logger.setLevel(logging.DEBUG)


class Results(NamedTuple):
    """
    NamedTuple with two fields: documents and scores. The `documents` field contains the
    retrieved documents or indices, while the `scores` field contains the scores of the
    retrieved documents or indices.
    """

    documents: np.ndarray
    scores: np.ndarray


def get_unique_tokens(
    corpus_tokens, show_progress=True, leave_progress=False, desc="Create Vocab"
):
    unique_tokens = set()
    for doc_tokens in tqdm(
        corpus_tokens, desc=desc, disable=not show_progress, leave=leave_progress
    ):
        unique_tokens.update(doc_tokens)
    return unique_tokens


def _calculate_scores_with_arrays(
    data, indptr, indices, num_docs, query_tokens_ids, dtype
):
    # First, we use the query_token_ids to select the relevant columns from the score_matrix
    query_tokens_ids = np.array(query_tokens_ids, dtype=int)
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores_lists = []
    indices_lists = []

    for i, (start, end) in enumerate(zip(indptr_starts, indptr_ends)):
        scores_lists.append(data[start:end])
        indices_lists.append(indices[start:end])

    # combine the lists into a single array

    scores = np.zeros(num_docs, dtype=dtype)
    if len(scores_lists) == 0:
        return scores

    scores_flat = np.concatenate(scores_lists)
    indices_flat = np.concatenate(indices_lists)
    np.add.at(scores, indices_flat, scores_flat)

    return scores


class BM25:
    def __init__(
        self,
        k1=1.5,
        b=0.75,
        delta=0.5,
        method="lucene",
        idf_method=None,
        dtype="float32",
        int_dtype="int32",
        corpus=None,
    ):
        """
        BM25S initialization.

        Parameters
        ----------
        k1 : float
            The k1 parameter in the BM25 formula.

        b : float
            The b parameter in the BM25 formula.

        delta : float
            The delta parameter in the BM25L and BM25+ formulas; it is ignored for other methods.

        method : str
            The method to use for scoring term frequency. Choose from 'robertson', 'lucene', 'atire'.

        idf_method : str
            The method to use for scoring inverse document frequency (same choices as `method`).
            If None, it will use the same method as `method`. If you are unsure, please do not
            change this parameter.
        dtype : str
            The data type of the BM25 scores.

        int_dtype : str
            The data type of the indices in the BM25 scores.

        corpus : Iterable[Dict]
            The corpus of documents. This is optional and is used for saving the corpus
            to the snapshot. We expect the corpus to be a list of dictionaries, where each
            dictionary represents a document.
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.dtype = dtype
        self.int_dtype = int_dtype
        self.method = method
        self.idf_method = idf_method if idf_method is not None else method
        self.methods_requiring_nonoccurrence = ("bm25l", "bm25+")
        self.corpus = corpus
        self._original_version = __version__

    @staticmethod
    def _infer_corpus_object(corpus):
        """
        Verifies if the corpus is a list of list of strings, an object with the `ids` and `vocab` attributes,
        or a tuple of two lists: first is list of list of ids, second is the vocab dictionary.
        """
        if hasattr(corpus, "ids") and hasattr(corpus, "vocab"):
            return "object"
        elif isinstance(corpus, tuple) and len(corpus) == 2:
            c1, c2 = corpus
            if isinstance(c1, list) and isinstance(c2, dict):
                return "tuple"
            else:
                raise ValueError(
                    "Corpus must be a list of list of tokens, an object with the `ids` and `vocab` attributes, or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document."
                )
        elif isinstance(corpus, Iterable):
            return "tokens"
        else:
            raise ValueError(
                "Corpus must be a list of list of tokens, an object with the `ids` and `vocab` attributes, or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document."
            )

    def build_index_from_ids(
        self,
        unique_token_ids: List[int],
        corpus_token_ids: List[List[int]],
        show_progress=True,
        leave_progress=False,
    ):
        """
        Low-level function to build the BM25 index from token IDs, used by the `index` method,
        as well as the `build_index_from_tokens` method.
        You can override this function if you want to build the index in a different way.

        Parameters
        ----------
        unique_token_ids : List[int]
            List of unique token IDs.

        corpus_token_ids : List[List[int]]
            List of list of token IDs for each document.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.
        """
        import scipy.sparse as sp

        avg_doc_len = np.array([len(doc_ids) for doc_ids in corpus_token_ids]).mean()
        n_docs = len(corpus_token_ids)
        n_vocab = len(unique_token_ids)

        # Step 1: Calculate the number of documents containing each token
        doc_frequencies = _calculate_doc_freqs(
            corpus_tokens=corpus_token_ids,
            unique_tokens=unique_token_ids,
            show_progress=show_progress,
            leave_progress=leave_progress,
        )

        # preliminary: if the method is one of BM25L or BM25+, we need to calculate the non-occurrence array
        if self.method in self.methods_requiring_nonoccurrence:
            self.nonoccurrence_array = _build_nonoccurrence_array(
                doc_frequencies=doc_frequencies,
                n_docs=n_docs,
                compute_idf_fn=_select_idf_scorer(self.idf_method),
                calculate_tfc_fn=_select_tfc_scorer(self.method),
                l_d=avg_doc_len,
                l_avg=avg_doc_len,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
                dtype=self.dtype,
            )
        else:
            self.nonoccurrence_array = None

        # Step 2: Calculate the idf for each token using the document frequencies
        idf_array = _build_idf_array(
            doc_frequencies=doc_frequencies,
            n_docs=n_docs,
            compute_idf_fn=_select_idf_scorer(self.idf_method),
            dtype=self.dtype,
        )

        # Step 3 Calculate the BM25 scores for each token in each document
        scores_flat, doc_idx, vocab_idx = _build_scores_and_indices_for_matrix(
            corpus_token_ids=corpus_token_ids,
            idf_array=idf_array,
            avg_doc_len=avg_doc_len,
            doc_frequencies=doc_frequencies,
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            show_progress=show_progress,
            leave_progress=leave_progress,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            method=self.method,
            nonoccurrence_array=self.nonoccurrence_array,
        )

        # Now, we build the sparse matrix
        score_matrix = sp.csc_matrix(
            (scores_flat, (doc_idx, vocab_idx)),
            shape=(n_docs, n_vocab),
            dtype=self.dtype,
        )
        data = score_matrix.data
        indices = score_matrix.indices
        indptr = score_matrix.indptr

        scores = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "num_docs": n_docs,
        }
        return scores

    def build_index_from_tokens(
        self, corpus_tokens, show_progress=True, leave_progress=False
    ):
        """
        Low-level function to build the BM25 index from tokens, used by the `index` method.
        You can override this function if you want to build the index in a different way.
        """
        unique_tokens = get_unique_tokens(
            corpus_tokens,
            show_progress=show_progress,
            leave_progress=leave_progress,
            desc="BM25S Create Vocab",
        )
        vocab_dict = {token: i for i, token in enumerate(unique_tokens)}
        unique_token_ids = [vocab_dict[token] for token in unique_tokens]

        corpus_token_ids = [
            [vocab_dict[token] for token in tokens]
            for tokens in tqdm(
                corpus_tokens,
                desc="BM25S Convert tokens to indices",
                leave=leave_progress,
                disable=not show_progress,
            )
        ]

        scores = self.build_index_from_ids(
            unique_token_ids=unique_token_ids,
            corpus_token_ids=corpus_token_ids,
            show_progress=show_progress,
            leave_progress=leave_progress,
        )

        return scores, vocab_dict

    def index(
        self,
        corpus: Union[Iterable, Tuple, tokenization.Tokenized],
        show_progress=True,
        leave_progress=False,
    ):
        """
        Given a `corpus` of documents, create the BM25 index. The `corpus` can be either:
        - An iterable of documents, where each document is a list of tokens (strings).
        - A tuple of two elements: the first is the list of unique token IDs (int), and the second is the vocabulary dictionary.
        - An object with the `ids` and `vocab` attributes, which are the unique token IDs and the token IDs for each document, respectively.

        Given a list of list of tokens, create the BM25 index.

        You can provide either the `corpus_tokens` or the `corpus_token_ids`. If you provide the `corpus_token_ids`,
        you must also provide the `vocab_dict` dictionary. If you provide the `corpus_tokens`, the vocab_dict
        dictionary will be created from the tokens, so you do not need to provide it.

        The `vocab_dict` dictionary is a mapping from tokens to their index in the vocabulary. This is used to
        create the sparse matrix representation of the BM25 scores, as well as during query time to convert the
        tokens to their indices.
        """
        inferred_corpus_obj = self._infer_corpus_object(corpus)

        if inferred_corpus_obj == "tokens":
            debug_logger.log(msg="Building index from tokens", level=logging.DEBUG)
            scores, vocab_dict = self.build_index_from_tokens(
                corpus, leave_progress=leave_progress, show_progress=show_progress
            )
        else:
            if inferred_corpus_obj == "tuple":
                debug_logger.log(msg="Building index from IDs", level=logging.DEBUG)
                corpus_token_ids, vocab_dict = corpus
            elif inferred_corpus_obj == "object":
                debug_logger.log(
                    msg="Building index from IDs objects", level=logging.DEBUG
                )
                corpus_token_ids = corpus.ids
                vocab_dict = corpus.vocab
            else:
                raise ValueError(
                    "Internal error: Found an invalid corpus object, indicating `_inferred_corpus_object` is not working correctly."
                )

            unique_token_ids = list(vocab_dict.values())
            scores = self.build_index_from_ids(
                unique_token_ids=unique_token_ids,
                corpus_token_ids=corpus_token_ids,
                leave_progress=leave_progress,
                show_progress=show_progress,
            )

        self.scores = scores
        self.vocab_dict = vocab_dict

    def get_tokens_ids(self, query_tokens: List[str]) -> List[int]:
        """
        For a given list of tokens, return the list of token IDs, leaving out tokens
        that are not in the vocabulary.
        """
        return [
            self.vocab_dict[token] for token in query_tokens if token in self.vocab_dict
        ]

    def get_scores_from_ids(self, query_tokens_ids: List[int]) -> np.ndarray:
        data = self.scores["data"]
        indices = self.scores["indices"]
        indptr = self.scores["indptr"]
        num_docs = self.scores["num_docs"]

        scores = _calculate_scores_with_arrays(
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            query_tokens_ids=query_tokens_ids,
            dtype=self.dtype,
        )

        # if there's a non-occurrence array, we need to add the non-occurrence score
        # back to the scores
        if self.nonoccurrence_array is not None:
            nonoccurrence_scores = self.nonoccurrence_array[query_tokens_ids].sum()
            scores += nonoccurrence_scores

        return scores

    def get_scores(self, query_tokens_single: List[str]) -> np.ndarray:
        query_tokens_ids = self.get_tokens_ids(query_tokens_single)
        return self.get_scores_from_ids(query_tokens_ids)

    def _get_top_k_results(
        self,
        query_tokens_single: List[str],
        k: int = 1000,
        backend="auto",
        sorted: bool = False,
    ):
        """
        This function is used to retrieve the top-k results for a single query.
        Since it's a hidden function, the user should not call it directly and
        may change in the future. Please use the `retrieve` function instead.
        """
        scores_q = self.get_scores(query_tokens_single)
        topk_scores, topk_indices = selection.topk(
            scores_q, k=k, sorted=sorted, backend=backend
        )

        return topk_scores, topk_indices

    def retrieve(
        self,
        query_tokens: List[List[str]],
        corpus: List[Any] = None,
        k: int = 10,
        sorted: bool = True,
        return_as: str = "tuple",
        show_progress: bool = True,
        leave_progress: bool = False,
        n_threads: int = 0,
        chunksize: int = 50,
        backend_selection: str = "auto",
    ):
        """
        Retrieve the top-k documents for each query (tokenized).

        Parameters
        ----------
        query_tokens : List[List[str]]
            List of list of tokens for each query. If a Tokenized object is provided,
            it will be converted to a list of list of tokens.

        corpus : List[str] or np.ndarray
            List of "documents" or a numpy array of documents. If provided, the function
            will return the documents instead of the indices. You do not have to provide
            the original documents (for example, you can provide the unique IDs of the
            documents here and then retrieve the actual documents from another source).

        k : int
            Number of documents to retrieve for each query.

        batch_size : int
            Number of queries to process in each batch. Internally, the function will
            process the queries in batches to speed up the computation.

        sorted : bool
            If True, the function will sort the results by score before returning them.

        return_as : str
            If return_as="tuple", a named tuple with two fields will be returned:
            `documents` and `scores`, which can be accessed as `result.documents` and
            `result.scores`, or by unpacking, e.g. `documents, scores = retrieve(...)`.
            If return_as="documents", only the retrieved documents (or indices if `corpus`
            is not provided) will be returned.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.
        
        leave_progress : bool
            If True, the progress bars will remain after the function completes.

        n_threads : int
            Number of jobs to run in parallel. If -1, it will use all available CPUs.
            If 0, it will run the jobs sequentially, without using multiprocessing.

        chunksize : int
            Number of batches to process in each job in the multiprocessing pool.

        backend_selection : str
            The backend to use for the top-k retrieval. Choose from "auto", "numpy", "jax".
            If "auto", it will use JAX if it is available, otherwise it will use numpy.
        """
        allowed_return_as = ["tuple", "documents"]

        if return_as not in allowed_return_as:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")
        else:
            pass

        if n_threads == -1:
            n_threads = os.cpu_count()

        if isinstance(query_tokens, tokenization.Tokenized):
            query_tokens = tokenization.convert_tokenized_to_string_list(query_tokens)

        tqdm_kwargs = {
            "total": len(query_tokens),
            "desc": "BM25S Retrieve",
            "leave": leave_progress,
            "disable": not show_progress,
        }
        topk_fn = partial(
            self._get_top_k_results, k=k, sorted=sorted, backend=backend_selection
        )

        if n_threads == 0:
            # Use a simple map function to retrieve the results
            out = tqdm(map(topk_fn, query_tokens), **tqdm_kwargs)
        else:
            # Use concurrent.futures.ProcessPoolExecutor to parallelize the computation
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                process_map = executor.map(
                    topk_fn,
                    query_tokens,
                    chunksize=chunksize,
                )
                out = list(tqdm(process_map, **tqdm_kwargs))

        scores, indices = zip(*out)
        scores, indices = np.array(scores), np.array(indices)

        corpus = corpus if corpus is not None else self.corpus

        if corpus is None:
            retrieved_docs = indices
        else:
            # if it is a JsonlCorpus object, we do not need to convert it to a list
            if isinstance(corpus, utils.corpus.JsonlCorpus):
                retrieved_docs = corpus[indices]
            else:
                index_flat = indices.flatten().tolist()
                results = [corpus[i] for i in index_flat]
                retrieved_docs = np.array(results).reshape(indices.shape)

        if return_as == "tuple":
            return Results(documents=retrieved_docs, scores=scores)
        elif return_as == "documents":
            return retrieved_docs
        else:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")

    def save(
        self,
        save_dir,
        corpus=None,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        nnoc_name="nonoccurrence_array.index.npy",
        corpus_name="corpus.jsonl",
        allow_pickle=False,
    ):
        """
        Save the BM25S index to the `save_dir` directory. This will save the scores array,
        the indices array, the indptr array, the vocab dictionary, and the parameters.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index will be saved.

        corpus : List[Dict]
            The corpus of documents. If provided, it will be saved to the `corpus` file.

        corpus_name : str
            The name of the file that will contain the corpus.

        data_name : str
            The name of the file that will contain the data array.

        indices_name : str
            The name of the file that will contain the indices array.

        indptr_name : str
            The name of the file that will contain the indptr array.

        vocab_name : str
            The name of the file that will contain the vocab dictionary.

        params_name : str
            The name of the file that will contain the parameters.

        nnoc_name : str
            The name of the file that will contain the non-occurrence array.

        allow_pickle : bool
            If True, the arrays will be saved using pickle. If False, the arrays will be saved
            in a more efficient format, but they will not be readable by older versions of numpy.
        """
        # Save the self.vocab_dict and self.score_matrix to the save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the scores arrays
        data_path = save_dir / data_name
        indices_path = save_dir / indices_name
        indptr_path = save_dir / indptr_name

        np.save(data_path, self.scores["data"], allow_pickle=allow_pickle)
        np.save(indices_path, self.scores["indices"], allow_pickle=allow_pickle)
        np.save(indptr_path, self.scores["indptr"], allow_pickle=allow_pickle)

        # save nonoccurrence array if it exists
        if self.nonoccurrence_array is not None:
            nnm_path = save_dir / nnoc_name
            np.save(nnm_path, self.nonoccurrence_array, allow_pickle=allow_pickle)

        # Save the vocab dictionary
        vocab_path = save_dir / vocab_name

        with open(vocab_path, "w") as f:
            json.dump(self.vocab_dict, f)

        # Save the parameters
        params_path = save_dir / params_name
        params = dict(
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            method=self.method,
            idf_method=self.idf_method,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            num_docs=self.scores["num_docs"],
            version=__version__,
        )
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)

        corpus = corpus if corpus is not None else self.corpus

        if corpus is not None:
            with open(save_dir / corpus_name, "w") as f:
                # if it's not an iterable, we skip
                if not isinstance(corpus, Iterable):
                    logging.warning(
                        "The corpus is not an iterable. Skipping saving the corpus."
                    )

                for i, doc in enumerate(corpus):
                    if isinstance(doc, str):
                        doc = {"id": i, "text": doc}
                    elif isinstance(doc, (dict, list, tuple)):
                        doc = doc
                    else:
                        logging.warning(
                            f"Document at index {i} is not a string, dictionary, list or tuple. Skipping."
                        )
                        continue

                    try:
                        doc = json.dumps(doc)
                    except Exception as e:
                        logging.warning(f"Error saving document at index {i}: {e}")
                    else:
                        f.write(doc + "\n")

            # also save corpus.mmindex
            mmidx = utils.corpus.find_newline_positions(save_dir / corpus_name)
            utils.corpus.save_mmindex(mmidx, path=save_dir / corpus_name)

    @classmethod
    def load(
        cls,
        save_dir,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        nnoc_name="nonoccurrence_array.index.npy",
        corpus_name="corpus.jsonl",
        load_corpus=False,
        mmap=False,
        allow_pickle=False,
    ):
        """
        Load a BM25S index that was saved using the `save` method.
        This returns a BM25S object with the saved parameters and scores,
        which can be directly used for retrieval.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index was saved.

        data_name : str
            The name of the file that contains the data array.

        indices_name : str
            The name of the file that contains the indices array.

        indptr_name : str
            The name of the file that contains the indptr array.

        vocab_name : str
            The name of the file that contains the vocab dictionary.

        params_name : str
            The name of the file that contains the parameters.

        nnoc_name : str
            The name of the file that contains the non-occurrence array.

        corpus_name : str
            The name of the file that contains the corpus.

        load_corpus : bool
            If True, the corpus will be loaded from the `corpus_name` file.

        mmap : bool
            Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
            If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very large arrays that
            do not fit into memory.

        allow_pickle : bool
            If True, the arrays will be loaded using pickle. If False, the arrays will be loaded
            in a more efficient format, but they will not be readable by older versions of numpy.
        """
        if not isinstance(mmap, bool):
            raise ValueError("`mmap` must be a boolean")

        # Load the BM25 index from the save_dir
        save_dir = Path(save_dir)

        # Load the parameters
        params_path = save_dir / params_name
        with open(params_path, "r") as f:
            params: dict = json.load(f)

        # Load the vocab dictionary
        vocab_path = save_dir / vocab_name
        with open(vocab_path, "r") as f:
            vocab_dict = json.load(f)

        # Load the score arrays
        data_path = save_dir / data_name
        indices_path = save_dir / indices_name
        indptr_path = save_dir / indptr_name

        mmap_mode = "r" if mmap else None
        data = np.load(data_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        indices = np.load(indices_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        indptr = np.load(indptr_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)

        original_version = params.pop("version", None)

        scores = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "num_docs": params.pop("num_docs", None),
        }


        bm25_obj = cls(**params)
        bm25_obj.scores = scores
        bm25_obj.vocab_dict = vocab_dict
        bm25_obj._original_version = original_version

        if load_corpus:
            # load the model from the snapshot
            # if a corpus.jsonl file exists, load it
            corpus_file = save_dir / corpus_name
            if os.path.exists(corpus_file):
                if mmap is True:
                    corpus = utils.corpus.JsonlCorpus(corpus_file)
                else:
                    corpus = []
                    with open(corpus_file, "r") as f:
                        for line in f:
                            doc = json.loads(line)
                            corpus.append(doc)

                bm25_obj.corpus = corpus

        # if the method is one of BM25L or BM25+, we need to load the non-occurrence array
        # if it does not exist, we raise an error
        if bm25_obj.method in bm25_obj.methods_requiring_nonoccurrence:
            nnm_path = save_dir / nnoc_name
            if nnm_path.exists():
                bm25_obj.nonoccurrence_array = np.load(
                    nnm_path, allow_pickle=allow_pickle
                )
            else:
                raise FileNotFoundError(f"Non-occurrence array not found at {nnm_path}")
        else:
            bm25_obj.nonoccurrence_array = None

        return bm25_obj
