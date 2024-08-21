from numba import njit, prange
import numpy as np
from typing import List
from ..scoring import _compute_relevance_from_scores_jit_ready
from .selection import _numba_sorted_top_k

_compute_relevance_from_scores_jit_ready = njit()(_compute_relevance_from_scores_jit_ready)

def _retrieve_internal_jittable(
    query_tokens_ids: List[List[str]],
    k: int,
    sorted: bool,
    dtype: np.dtype,
    int_dtype: np.dtype,
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    nonoccurrence_array: np.ndarray = None,
):
    topk_scores = np.zeros((len(query_tokens_ids), k), dtype=dtype)
    topk_indices = np.zeros((len(query_tokens_ids), k), dtype=int_dtype)

    for i in range(len(query_tokens_ids)):
        query_tokens_single = query_tokens_ids[i]
        query_tokens_single = np.asarray(query_tokens_single, dtype=int_dtype)
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

        topk_scores_sing, topk_indices_sing = _numba_sorted_top_k(
            scores_single, k=k, sorted=sorted
        )
        topk_scores[i] = topk_scores_sing
        topk_indices[i] = topk_indices_sing

    return topk_scores, topk_indices