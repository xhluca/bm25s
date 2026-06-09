import os
from numba import njit, prange, get_num_threads, get_thread_id
import numpy as np
from typing import List, Tuple, Any
import logging

from .. import utils
from ..scoring import _compute_relevance_from_scores_jit_ready
from .selection import _numba_sorted_top_k, heap_push, sift_up

_compute_relevance_from_scores_jit_ready = njit()(_compute_relevance_from_scores_jit_ready)

@njit(parallel=True)
def _retrieve_internal_jitted_parallel(
    query_tokens_ids_flat: np.ndarray,
    query_pointers: np.ndarray,
    k: int,
    sorted: bool,
    dtype: np.dtype,
    int_dtype: np.dtype,
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    weight_mask: np.ndarray = None,
):
    N = len(query_pointers) - 1
    n_threads = get_num_threads()

    topk_scores = np.zeros((N, k), dtype=dtype)
    topk_indices = np.zeros((N, k), dtype=int_dtype)

    # Each thread reuses a dense score accumulator along with the list of
    # candidate documents touched by the current query. Tracking candidates
    # lets us run the top-k selection over the touched documents only, and
    # reset just those entries afterwards, instead of zeroing and scanning
    # all `num_docs` scores for every query.
    scores_threads = np.zeros((n_threads, num_docs), dtype=dtype)
    candidates_threads = np.empty((n_threads, num_docs), dtype=np.int32)

    for i in prange(N):
        tid = get_thread_id()
        scores = scores_threads[tid]
        candidates = candidates_threads[tid]

        # When the posting lists of the query are long, most documents are
        # touched anyway and the candidate bookkeeping costs more than the
        # full scan it avoids, so fall back to scanning the dense scores.
        total_postings = 0
        for q_ptr in range(query_pointers[i], query_pointers[i + 1]):
            t = query_tokens_ids_flat[q_ptr]
            total_postings += indptr[t + 1] - indptr[t]
        track_candidates = total_postings < num_docs // 2

        n_candidates = 0
        for q_ptr in range(query_pointers[i], query_pointers[i + 1]):
            t = query_tokens_ids_flat[q_ptr]
            if track_candidates:
                for j in range(indptr[t], indptr[t + 1]):
                    d = indices[j]
                    v = data[j]
                    # BM25 scores are nonnegative, so a document's score is
                    # zero if and only if the query has not touched it yet
                    if v != 0:
                        if scores[d] == 0:
                            candidates[n_candidates] = d
                            n_candidates += 1
                        scores[d] += v
            else:
                for j in range(indptr[t], indptr[t + 1]):
                    scores[indices[j]] += data[j]

        # Top-k selection with a min-heap over the candidate documents (or all
        # documents in the dense case), written directly into the output row
        # of this query. Visited scores are reset as they are read, leaving
        # the accumulator clean for the next query of this thread -- except
        # when there are fewer candidates than k, where the scores are needed
        # to pad the results with untouched (zero-score) documents first.
        values = topk_scores[i]
        inds = topk_indices[i]
        length = 0
        needs_padding = track_candidates and n_candidates < k
        scan_size = n_candidates if track_candidates else num_docs
        for c in range(scan_size):
            d = candidates[c] if track_candidates else c
            v = scores[d]
            if v != 0 and not needs_padding:
                scores[d] = 0
            if weight_mask is not None:
                v = v * weight_mask[d]
            if length < k:
                heap_push(values, inds, v, d, length)
                length += 1
            elif v > values[0]:
                values[0] = v
                inds[0] = d
                sift_up(values, inds, 0, length)

        if needs_padding:
            # documents with a zero score were not touched by the query, so
            # they cannot already be among the candidates in the heap
            d = 0
            while length < k:
                if scores[d] == 0:
                    values[length] = 0
                    inds[length] = d
                    length += 1
                d += 1
            for c in range(n_candidates):
                scores[candidates[c]] = 0

        if sorted:
            sorted_inds = np.flip(np.argsort(values))
            topk_scores[i] = values[sorted_inds]
            topk_indices[i] = inds[sorted_inds]

    return topk_scores, topk_indices


@njit(parallel=True)
def _retrieve_internal_jitted_parallel_nonoccurrence(
    query_tokens_ids_flat: np.ndarray,
    query_pointers: np.ndarray,
    k: int,
    sorted: bool,
    dtype: np.dtype,
    int_dtype: np.dtype,
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    nonoccurrence_array: np.ndarray = None,
    weight_mask: np.ndarray = None,
):
    # The nonoccurrence score (used by BM25L and BM25+) shifts the score of
    # every document in the corpus, so the dense scores are computed and
    # scanned in full rather than tracking the documents touched by the query.
    N = len(query_pointers) - 1

    topk_scores = np.zeros((N, k), dtype=dtype)
    topk_indices = np.zeros((N, k), dtype=int_dtype)

    for i in prange(N):
        query_tokens_single = query_tokens_ids_flat[query_pointers[i] : query_pointers[i + 1]]

        scores_single = _compute_relevance_from_scores_jit_ready(
            query_tokens_ids=query_tokens_single,
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            dtype=dtype,
        )

        # if there's a non-occurrence array, we need to add the non-occurrence score
        # back to the scores
        if nonoccurrence_array is not None:
            nonoccurrence_scores = nonoccurrence_array[query_tokens_single].sum()
            scores_single += nonoccurrence_scores

        if weight_mask is not None:
            scores_single = scores_single * weight_mask

        topk_scores_sing, topk_indices_sing = _numba_sorted_top_k(
            scores_single, k=k, sorted=sorted
        )
        topk_scores[i] = topk_scores_sing
        topk_indices[i] = topk_indices_sing

    return topk_scores, topk_indices


def _retrieve_numba_functional(
    query_tokens_ids,
    scores,
    corpus: List[Any] = None,
    k: int = 10,
    sorted: bool = True,
    return_as: str = "tuple",
    show_progress: bool = True,
    leave_progress: bool = False,
    n_threads: int = 0,
    chunksize: int = None,
    nonoccurrence_array=None,
    backend_selection="numba",
    dtype="float32",
    int_dtype="int32",
    weight_mask=None,
):
    from numba import get_num_threads, set_num_threads, njit


    if backend_selection != "numba":
        error_msg = "The `numba` backend must be selected when retrieving using the numba backend. Please choose a different backend or change the backend_selection parameter to numba."
        raise ValueError(error_msg)

    if chunksize != None:
        # warn the user that the chunksize parameter is ignored
        logging.warning(
            "The `chunksize` parameter is ignored in the `retrieve` function when using the `numba` backend."
            "The function will automatically determine the best chunksize."
        )

    allowed_return_as = ["tuple", "documents"]

    if return_as not in allowed_return_as:
        raise ValueError("`return_as` must be either 'tuple' or 'documents'")
    else:
        pass

    if n_threads == -1:
        n_threads = os.cpu_count()
    elif n_threads == 0:
        n_threads = 1

    # get og thread count
    og_n_threads = get_num_threads()
    set_num_threads(n_threads)


    # convert query_tokens_ids from list of list to a flat 1-d np.ndarray with
    # pointers to the start of each query to be used to find the boundaries of each query
    query_pointers = np.cumsum([0] + [len(q) for q in query_tokens_ids], dtype=int_dtype)
    query_tokens_ids_flat = np.concatenate(query_tokens_ids).astype(int_dtype)

    if nonoccurrence_array is None:
        retrieved_scores, retrieved_indices = _retrieve_internal_jitted_parallel(
            query_pointers=query_pointers,
            query_tokens_ids_flat=query_tokens_ids_flat,
            k=k,
            sorted=sorted,
            dtype=np.dtype(dtype),
            int_dtype=np.dtype(int_dtype),
            data=scores["data"],
            indptr=scores["indptr"],
            indices=scores["indices"],
            num_docs=scores["num_docs"],
            weight_mask=weight_mask,
        )
    else:
        retrieved_scores, retrieved_indices = _retrieve_internal_jitted_parallel_nonoccurrence(
            query_pointers=query_pointers,
            query_tokens_ids_flat=query_tokens_ids_flat,
            k=k,
            sorted=sorted,
            dtype=np.dtype(dtype),
            int_dtype=np.dtype(int_dtype),
            data=scores["data"],
            indptr=scores["indptr"],
            indices=scores["indices"],
            num_docs=scores["num_docs"],
            nonoccurrence_array=nonoccurrence_array,
            weight_mask=weight_mask,
        )

    # reset the number of threads
    set_num_threads(og_n_threads)

    if corpus is None:
        retrieved_docs = retrieved_indices
    else:
        # if it is a JsonlCorpus object, we do not need to convert it to a list
        if isinstance(corpus, utils.corpus.JsonlCorpus):
            retrieved_docs = corpus[retrieved_indices]
        elif isinstance(corpus, np.ndarray) and corpus.ndim == 1:
            retrieved_docs = corpus[retrieved_indices]
        else:
            index_flat = retrieved_indices.flatten().tolist()
            results = [corpus[i] for i in index_flat]
            retrieved_docs = np.array(results).reshape(retrieved_indices.shape)

    if return_as == "tuple":
        return retrieved_docs, retrieved_scores
    elif return_as == "documents":
        return retrieved_docs
    else:
        raise ValueError("`return_as` must be either 'tuple' or 'documents'")
