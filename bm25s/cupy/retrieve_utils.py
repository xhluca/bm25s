import logging
from typing import Any, List

import numpy as np

from .. import utils
from .selection import _require_cupy, topk


def _score_query_cupy(
    query_tokens_ids,
    data,
    indices,
    indptr,
    num_docs,
    dtype,
    nonoccurrence_array=None,
    weight_mask=None,
):
    cupy = _require_cupy()
    scores = cupy.zeros(num_docs, dtype=dtype)

    for token_id in query_tokens_ids:
        token_id = int(token_id)
        start = int(indptr[token_id])
        end = int(indptr[token_id + 1])
        if end > start:
            cupy.add.at(scores, indices[start:end], data[start:end])

    if weight_mask is not None:
        scores *= weight_mask

    if nonoccurrence_array is not None and len(query_tokens_ids) > 0:
        query_tokens_ids_gpu = cupy.asarray(query_tokens_ids, dtype=np.int64)
        scores += nonoccurrence_array[query_tokens_ids_gpu].sum()

    return scores


def _retrieve_cupy_functional(
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
    backend_selection="cupy",
    dtype="float32",
    int_dtype="int32",
    weight_mask=None,
):
    cupy = _require_cupy()

    if backend_selection == "auto":
        backend_selection = "cupy"

    if backend_selection != "cupy":
        error_msg = (
            "The `cupy` backend must use backend_selection='cupy' or 'auto'. "
            "Please choose a different backend or change backend_selection to cupy."
        )
        raise ValueError(error_msg)

    if chunksize is not None:
        logging.warning(
            "The `chunksize` parameter is ignored in the `retrieve` function "
            "when using the `cupy` backend."
        )

    if n_threads not in (0, 1):
        logging.warning(
            "The `n_threads` parameter is ignored in the `retrieve` function "
            "when using the `cupy` backend."
        )

    allowed_return_as = ["tuple", "documents"]
    if return_as not in allowed_return_as:
        raise ValueError("`return_as` must be either 'tuple' or 'documents'")

    dtype = np.dtype(dtype)
    int_dtype = np.dtype(int_dtype)
    num_queries = len(query_tokens_ids)

    data_gpu = cupy.asarray(scores["data"], dtype=dtype)
    indices_gpu = cupy.asarray(scores["indices"], dtype=int_dtype)
    indptr = np.asarray(scores["indptr"], dtype=int_dtype)

    if nonoccurrence_array is not None:
        nonoccurrence_array_gpu = cupy.asarray(nonoccurrence_array, dtype=dtype)
    else:
        nonoccurrence_array_gpu = None

    if weight_mask is not None:
        weight_mask_gpu = cupy.asarray(weight_mask)
    else:
        weight_mask_gpu = None

    retrieved_scores = np.empty((num_queries, k), dtype=dtype)
    retrieved_indices = np.empty((num_queries, k), dtype=int_dtype)

    for i, query_tokens_ids_single in enumerate(query_tokens_ids):
        query_tokens_ids_single = np.asarray(query_tokens_ids_single, dtype=int_dtype)
        scores_single = _score_query_cupy(
            query_tokens_ids=query_tokens_ids_single,
            data=data_gpu,
            indices=indices_gpu,
            indptr=indptr,
            num_docs=scores["num_docs"],
            dtype=dtype,
            nonoccurrence_array=nonoccurrence_array_gpu,
            weight_mask=weight_mask_gpu,
        )

        topk_scores_sing, topk_indices_sing = topk(
            scores_single, k=k, backend="cupy", sorted=sorted
        )
        retrieved_scores[i] = topk_scores_sing.astype(dtype, copy=False)
        retrieved_indices[i] = topk_indices_sing.astype(int_dtype, copy=False)

    if corpus is None:
        retrieved_docs = retrieved_indices
    else:
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
