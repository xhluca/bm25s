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
    n_scores = int(query_scores_gpu.shape[0])
    if k > n_scores:
        k = n_scores

    if k == 0:
        scores_dtype = np.asarray(query_scores).dtype
        return np.empty(0, dtype=scores_dtype), np.empty(0, dtype=np.int64)

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

    return cupy.asnumpy(query_scores_gpu), cupy.asnumpy(ind)


def topk(query_scores, k, backend="cupy", sorted=True):
    """
    Retrieve the top-k results for a single 1-dimensional score array with CuPy.
    """
    if backend != "cupy":
        raise ValueError("Invalid backend. Only 'cupy' is supported.")

    return _topk_cupy(query_scores, k, sorted)
