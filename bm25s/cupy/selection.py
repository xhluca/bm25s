import os

import numpy as np


try:
    import cupy as cp
except Exception:
    cp = None
    CUPY_AVAILABLE = False
else:
    CUPY_AVAILABLE = True


def _require_cupy():
    if cp is None:
        raise ImportError(
            "CuPy is not installed. Please install a CuPy package compatible with "
            "your CUDA runtime to use the cupy backend."
        )
    return cp


def _topk_cupy(query_scores, k, sorted):
    cupy = _require_cupy()

    query_scores_gpu = cupy.asarray(query_scores)
    query_scores_gpu, ind = _topk_cupy_gpu(query_scores_gpu, k, sorted)
    return cupy.asnumpy(query_scores_gpu), cupy.asnumpy(ind)


def _topk_cupy_gpu(query_scores_gpu, k, sorted):
    cupy = _require_cupy()

    query_scores_gpu = cupy.asarray(query_scores_gpu)
    n_scores = int(query_scores_gpu.shape[0])
    if k > n_scores:
        k = n_scores

    if k == 0:
        return (
            cupy.empty(0, dtype=query_scores_gpu.dtype),
            cupy.empty(0, dtype=cupy.int64),
        )

    if os.environ.get("BM25S_CUPY_TOPK_MODE", "sort") == "sort":
        return _topk_cupy_sort_gpu(query_scores_gpu, k, sorted)

    return _topk_cupy_partition_gpu(query_scores_gpu, k, sorted)


def _topk_cupy_sort_gpu(query_scores_gpu, k, sorted):
    cupy = _require_cupy()

    order = cupy.argsort(query_scores_gpu)
    ind = order[-k:]
    if sorted:
        ind = ind[::-1]

    query_scores_gpu = cupy.take(query_scores_gpu, ind)
    return query_scores_gpu, ind


def _topk_cupy_partition_gpu(query_scores_gpu, k, sorted):
    cupy = _require_cupy()

    partitioned_ind = cupy.argpartition(query_scores_gpu, -k)
    partitioned_ind = partitioned_ind.take(indices=range(-k, 0))
    partitioned_scores = cupy.take(query_scores_gpu, partitioned_ind)

    if sorted:
        sorted_trunc_ind = cupy.flip(cupy.argsort(partitioned_scores))
        ind = partitioned_ind[sorted_trunc_ind]
        query_scores_gpu = partitioned_scores[sorted_trunc_ind]
    else:
        ind = partitioned_ind
        query_scores_gpu = partitioned_scores

    return query_scores_gpu, ind


def topk(query_scores, k, backend="cupy", sorted=True):
    """
    Retrieve the top-k results for a single 1-dimensional score array with CuPy.
    """
    if backend != "cupy":
        raise ValueError("Invalid backend. Only 'cupy' is supported.")

    return _topk_cupy(query_scores, k, sorted)
