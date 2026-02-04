import os
from numba import njit, prange
import numpy as np
from typing import List, Tuple, Any
import logging

from .. import utils
from ..scoring import _compute_relevance_from_scores_jit_ready
from .selection import _numba_sorted_top_k

_compute_relevance_from_scores_jit_ready = njit()(_compute_relevance_from_scores_jit_ready)


@njit
def _fused_score_and_topk(
    query_tokens,
    data,
    indptr,
    indices,
    num_docs,
    k,
    dtype,
    nonoccurrence_array=None,
    weight_mask=None,
    sorted_results=True,
):
    """
    Fused scoring and top-k selection that avoids iterating through all documents.
    Only documents with non-zero scores are considered for top-k selection.
    This provides significant speedup when num_docs >> number of matching docs.
    """
    # Allocate score array and mask for tracking non-zero docs
    scores = np.zeros(num_docs, dtype=dtype)
    doc_mask = np.zeros(num_docs, dtype=np.bool_)

    # Pre-compute total number of entries to size the buffer
    total_entries = 0
    for token_id in query_tokens:
        total_entries += indptr[token_id + 1] - indptr[token_id]

    # Buffer to collect unique doc indices during scoring
    seen_docs_buffer = np.empty(total_entries, dtype=np.int32)
    buf_idx = 0

    # Compute scores and collect unique doc indices in one pass
    for token_id in query_tokens:
        start = indptr[token_id]
        end = indptr[token_id + 1]
        for j in range(start, end):
            doc_idx = indices[j]
            scores[doc_idx] += data[j]
            if not doc_mask[doc_idx]:
                doc_mask[doc_idx] = True
                seen_docs_buffer[buf_idx] = doc_idx
                buf_idx += 1

    nonzero_count = buf_idx

    # Handle nonoccurrence array (for bm25l, bm25+ variants)
    if nonoccurrence_array is not None:
        nonoccurrence_sum = np.float32(0.0)
        for token_id in query_tokens:
            nonoccurrence_sum += nonoccurrence_array[token_id]
        # Add to all scored documents
        for i in range(nonzero_count):
            scores[seen_docs_buffer[i]] += nonoccurrence_sum

    # Handle weight mask
    if weight_mask is not None:
        for i in range(nonzero_count):
            doc_idx = seen_docs_buffer[i]
            scores[doc_idx] *= weight_mask[doc_idx]

    # Top-k selection on only the non-zero documents
    actual_k = min(k, nonzero_count)

    # Use a min-heap for top-k
    heap_values = np.zeros(actual_k, dtype=dtype)
    heap_indices = np.zeros(actual_k, dtype=np.int32)
    heap_size = 0

    for i in range(nonzero_count):
        doc_idx = seen_docs_buffer[i]
        score = scores[doc_idx]

        if heap_size < actual_k:
            # Add to heap
            heap_values[heap_size] = score
            heap_indices[heap_size] = doc_idx
            heap_size += 1
            # Sift up (bubble up)
            pos = heap_size - 1
            while pos > 0:
                parent = (pos - 1) >> 1
                if heap_values[pos] < heap_values[parent]:
                    heap_values[pos], heap_values[parent] = heap_values[parent], heap_values[pos]
                    heap_indices[pos], heap_indices[parent] = heap_indices[parent], heap_indices[pos]
                    pos = parent
                else:
                    break
        elif score > heap_values[0]:
            # Replace min element and sift down
            heap_values[0] = score
            heap_indices[0] = doc_idx
            pos = 0
            while True:
                left = 2 * pos + 1
                right = 2 * pos + 2
                smallest = pos
                if left < heap_size and heap_values[left] < heap_values[smallest]:
                    smallest = left
                if right < heap_size and heap_values[right] < heap_values[smallest]:
                    smallest = right
                if smallest != pos:
                    heap_values[pos], heap_values[smallest] = heap_values[smallest], heap_values[pos]
                    heap_indices[pos], heap_indices[smallest] = heap_indices[smallest], heap_indices[pos]
                    pos = smallest
                else:
                    break

    # Sort results if requested (descending by score)
    if sorted_results:
        sorted_order = np.argsort(heap_values)[::-1]
        result_scores = heap_values[sorted_order]
        result_indices = heap_indices[sorted_order]
    else:
        result_scores = heap_values
        result_indices = heap_indices

    # Pad with zeros if needed (when nonzero_count < k)
    if actual_k < k:
        full_scores = np.zeros(k, dtype=dtype)
        full_indices = np.zeros(k, dtype=np.int32)
        full_scores[:actual_k] = result_scores
        full_indices[:actual_k] = result_indices
        return full_scores, full_indices

    return result_scores, result_indices

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
    nonoccurrence_array: np.ndarray = None,
    weight_mask: np.ndarray = None,
):
    N = len(query_pointers) - 1

    topk_scores = np.zeros((N, k), dtype=dtype)
    topk_indices = np.zeros((N, k), dtype=int_dtype)

    for i in prange(N):
        query_tokens_single = query_tokens_ids_flat[query_pointers[i] : query_pointers[i + 1]]

        # Use fused scoring + top-k for better performance
        topk_scores_sing, topk_indices_sing = _fused_score_and_topk(
            query_tokens=query_tokens_single,
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            k=k,
            dtype=dtype,
            nonoccurrence_array=nonoccurrence_array,
            weight_mask=weight_mask,
            sorted_results=sorted,
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
