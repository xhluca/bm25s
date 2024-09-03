"""
Acknowledgement:
numba_unsorted_top_k is taken from retriv. The original code can be found at:
https://github.com/AmenRa/retriv/blob/v0.2.1/retriv/utils/numba_utils.py

numba_sorted_top_k was created based on numba_unsorted_top_k, but modified to use a heap to keep track of the top-k values.
"""

import numpy as np
from numba import njit


@njit()
def _numba_unsorted_top_k_legacy(array: np.ndarray, k: int):
    top_k_values = np.zeros(k, dtype=np.float32)
    top_k_indices = np.zeros(k, dtype=np.int32)

    min_value = 0.0
    min_value_idx = 0

    for i, value in enumerate(array):
        if value > min_value:
            top_k_values[min_value_idx] = value
            top_k_indices[min_value_idx] = i
            min_value_idx = top_k_values.argmin()
            min_value = top_k_values[min_value_idx]

    return top_k_values, top_k_indices


@njit()
def sift_down(values, indices, startpos, pos):
    new_value = values[pos]
    new_index = indices[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent_value = values[parentpos]
        if new_value < parent_value:
            values[pos] = parent_value
            indices[pos] = indices[parentpos]
            pos = parentpos
            continue
        break
    values[pos] = new_value
    indices[pos] = new_index


@njit()
def sift_up(values, indices, pos, length):
    startpos = pos
    new_value = values[pos]
    new_index = indices[pos]
    childpos = 2 * pos + 1
    while childpos < length:
        rightpos = childpos + 1
        if rightpos < length and values[rightpos] < values[childpos]:
            childpos = rightpos
        values[pos] = values[childpos]
        indices[pos] = indices[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    values[pos] = new_value
    indices[pos] = new_index
    sift_down(values, indices, startpos, pos)


@njit()
def heap_push(values, indices, value, index, length):
    values[length] = value
    indices[length] = index
    sift_down(values, indices, 0, length)


@njit()
def heap_pop(values, indices, length):
    return_value = values[0]
    return_index = indices[0]
    last_value = values[length - 1]
    last_index = indices[length - 1]
    values[0] = last_value
    indices[0] = last_index
    sift_up(values, indices, 0, length - 1)
    return return_value, return_index


@njit()
def _numba_sorted_top_k(array: np.ndarray, k: int, sorted=True):
    n = len(array)
    if k > n:
        k = n

    values = np.zeros(k, dtype=array.dtype)  # aka scores
    indices = np.zeros(k, dtype=np.int32)
    length = 0

    for i, value in enumerate(array):
        if length < k:
            heap_push(values, indices, value, i, length)
            length += 1
        else:
            if value > values[0]:
                values[0] = value
                indices[0] = i
                sift_up(values, indices, 0, length)

    if sorted:
        # # This is the original code for sorting, we can skip it and return the values and indices
        # # to let numpy handle the sorting
        # top_k_values = np.zeros(k, dtype=array.dtype)
        # top_k_indices = np.zeros(k, dtype=np.int32)

        # for i in range(k - 1, -1, -1):
        #     top_k_values[i], top_k_indices[i] = heap_pop(values, indices, length)
        #     length -= 1
        # values = top_k_values
        # indices = top_k_indices

        # This is the new code that uses numpy to sort the values and indices instead of 
        # using the heap to sort them.
        sorted_indices = np.flip(np.argsort(values))
        indices = indices[sorted_indices]
        values = values[sorted_indices]

    return values, indices


def topk(query_scores, k, backend="numba", sorted=True):
    """
    This function is used to retrieve the top-k results for a single query. It will only work
    on a 1-dimensional array of scores.
    """
    if backend not in ["numba"]:
        raise ValueError(
            "Invalid backend. Only 'numba' is supported."
        )
    elif backend == "numba":
        uns_scores, uns_indices = _numba_sorted_top_k(query_scores, k)
        if sorted:
            sorted_inds = np.flip(np.argsort(uns_scores))
            query_inds = uns_indices[sorted_inds]
            query_scores = uns_scores[sorted_inds]
        else:
            query_inds = uns_indices
            query_scores = uns_scores
        
        return query_scores, query_inds

    else:
        raise ValueError("Invalid backend. Only 'numba' is supported.")
