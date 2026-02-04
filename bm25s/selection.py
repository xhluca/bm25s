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
    # np.argpartition is faster than np.argsort, but does not return values in order
    # Use slicing [-k:] instead of .take(range(-k, 0)) for better performance
    partitioned_ind = np.argpartition(query_scores, -k)[-k:]

    if sorted:
        # Sort by scores at the partitioned indices (descending)
        order = np.argsort(query_scores[partitioned_ind])[::-1]
        ind = partitioned_ind[order]
        scores = query_scores[ind]
    else:
        ind = partitioned_ind
        scores = query_scores[partitioned_ind]

    return scores, ind


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
