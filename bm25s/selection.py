import numpy as np

try:
    import jax.lax
except ImportError:
    JAX_IS_AVAILABLE = False
else:
    JAX_IS_AVAILABLE = True
    # if JAX is available, we need to initialize it with a dummy scores and capture
    # any output to avoid it from saying that gpu is not available
    _ = jax.lax.top_k(np.array([0] * 5), 1)


def _topk_numpy(query_scores, k, sorted):
    # https://stackoverflow.com/questions/65038206/how-to-get-indices-of-top-k-values-from-a-numpy-array
    # np.argpartition is faster than np.argsort, but do not return the values in order
    partitioned_ind = np.argpartition(query_scores, -k)
    # Since lit's a single query, we can take the last k elements
    partitioned_ind = partitioned_ind.take(indices=range(-k, 0))
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take(query_scores, partitioned_ind)

    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(np.argsort(partitioned_scores))

        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = partitioned_ind[sorted_trunc_ind]
        query_scores = partitioned_scores[sorted_trunc_ind]

    else:
        ind = partitioned_ind
        query_scores = partitioned_scores

    return query_scores, ind


def _topk_jax(query_scores, k):
    topk_scores, topk_indices = jax.lax.top_k(query_scores, k)
    topk_scores = np.asarray(topk_scores)
    topk_indices = np.asarray(topk_indices)

    return topk_scores, topk_indices


def topk(query_scores, k, backend="auto", sorted=True):
    """
    This function is used to retrieve the top-k results for a single query. It will only work
    on a 1-dimensional array of scores.
    """
    if backend == "auto":
        # if jax.lax is available, use it to speed up selection, otherwise use numpy
        backend = "jax" if JAX_IS_AVAILABLE else "numpy"
    
    if backend not in ["numpy", "jax"]:
        raise ValueError("Invalid backend. Please choose from 'numpy' or 'jax'.")
    elif backend == "jax":
        if not JAX_IS_AVAILABLE:
            raise ImportError("JAX is not available. Please install JAX with `pip install jax[cpu]` to use this backend.")
        return _topk_jax(query_scores, k)
    else:
        return _topk_numpy(query_scores, k, sorted)
