import os
from numba import njit, prange
import numpy as np
from typing import List, Tuple, Any
import logging

from .. import utils
from ..scoring import _compute_relevance_from_scores_jit_ready
from .selection import _numba_sorted_top_k, heap_push, sift_up

_compute_relevance_from_scores_jit_ready = njit(cache=True)(_compute_relevance_from_scores_jit_ready)

# Documents are grouped in chunks of this size during top-k selection; the
# maximum score of each chunk is computed first, so that entire chunks that
# cannot contain a top-k document are skipped in the detailed pass.
_SELECTION_CHUNK_SIZE = 512

# Type signatures for which the exhaustive selection kernel has already been
# compiled (see _retrieve_numba_functional)
_EXHAUSTIVE_KERNEL_WARMED = set()


@njit(parallel=True, cache=True)
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
    exact_ties: bool = False,
    weight_mask: np.ndarray = None,
):
    N = len(query_pointers) - 1
    chunk = _SELECTION_CHUNK_SIZE
    n_chunks = (num_docs + chunk - 1) // chunk

    topk_scores = np.zeros((N, k), dtype=dtype)
    topk_indices = np.zeros((N, k), dtype=int_dtype)
    n_exhaustive = 0

    for i in prange(N):
        scores = np.zeros(num_docs, dtype=dtype)
        for q_ptr in range(query_pointers[i], query_pointers[i + 1]):
            t = query_tokens_ids_flat[q_ptr]
            for j in range(indptr[t], indptr[t + 1]):
                scores[indices[j]] += data[j]

        values = topk_scores[i]
        inds = topk_indices[i]
        length = 0
        tau = np.float32(-1.0)  # the heap root once the heap is full

        if n_chunks > k:
            # Pass 1: maximum score of every chunk of documents, with a
            # branchless unrolled reduction (four independent max chains)
            chunk_maxes = np.empty(n_chunks, dtype=dtype)
            for c in range(n_chunks):
                start = c * chunk
                end = min(start + chunk, num_docs)
                m0 = np.float32(0.0)
                m1 = np.float32(0.0)
                m2 = np.float32(0.0)
                m3 = np.float32(0.0)
                x = start
                if weight_mask is None:
                    while x + 4 <= end:
                        m0 = max(m0, scores[x])
                        m1 = max(m1, scores[x + 1])
                        m2 = max(m2, scores[x + 2])
                        m3 = max(m3, scores[x + 3])
                        x += 4
                    while x < end:
                        m0 = max(m0, scores[x])
                        x += 1
                else:
                    while x < end:
                        m0 = max(m0, scores[x] * weight_mask[x])
                        x += 1
                chunk_maxes[c] = max(max(m0, m1), max(m2, m3))

            # Pass 2: every chunk max is the (masked) score of one real
            # document, and the chunk maxes cover pairwise distinct documents,
            # so the kth largest chunk max is a valid lower bound on the kth
            # best score. Priming the selection threshold with it lets the
            # detailed pass skip most chunks outright, instead of warming the
            # threshold up from zero while scanning every document.
            bound_values = np.empty(k, dtype=dtype)
            bound_inds = np.empty(k, dtype=int_dtype)
            bound_length = 0
            for c in range(n_chunks):
                v = chunk_maxes[c]
                if bound_length < k:
                    heap_push(bound_values, bound_inds, v, c, bound_length)
                    bound_length += 1
                elif v > bound_values[0]:
                    bound_values[0] = v
                    sift_up(bound_values, bound_inds, 0, bound_length)
            fill_bound = bound_values[0]

            # Pass 3: top-k selection with a min-heap over the chunks that can
            # still contain a top-k document. At least k documents score at
            # least fill_bound (see above), so the heap is guaranteed to fill.
            # Documents whose score ties the selection threshold are tracked:
            # which of the tied documents survive (and how equal scores are
            # ordered) depends on the order in which documents enter the heap,
            # so those queries are redone below with the exhaustive scan to
            # return exactly what it would have returned.
            tie_possible = False
            for c in range(n_chunks):
                if length >= k:
                    if chunk_maxes[c] <= tau:
                        if chunk_maxes[c] == tau:
                            tie_possible = True
                        continue
                elif chunk_maxes[c] < fill_bound:
                    continue
                start = c * chunk
                end = min(start + chunk, num_docs)
                for d in range(start, end):
                    v = scores[d]
                    if weight_mask is not None:
                        v = v * weight_mask[d]
                    if length < k:
                        if v >= fill_bound:
                            heap_push(values, inds, v, d, length)
                            length += 1
                            if length == k:
                                tau = values[0]
                    elif v > tau:
                        evicted = values[0]
                        values[0] = v
                        inds[0] = d
                        sift_up(values, inds, 0, length)
                        tau = values[0]
                        if evicted == tau:
                            # a document tied with the evicted one remains:
                            # which of the two was dropped depends on the
                            # heap layout, so fall back to the exact scan
                            tie_possible = True
                    elif v == tau:
                        tie_possible = True

            # equal scores anywhere inside the top-k also make the final
            # ordering depend on the traversal, so check for duplicates
            if exact_ties and not tie_possible:
                check_order = np.argsort(values)
                for x in range(k - 1):
                    if values[check_order[x]] == values[check_order[x + 1]]:
                        tie_possible = True
                        break
            exhaustive = exact_ties and tie_possible
        else:
            # k is at least the number of chunks, so the chunk-max bound is
            # vacuous and the exhaustive scan is used directly
            exhaustive = True

        if exhaustive:
            # Plain single pass over all scores, identical (heap operation by
            # heap operation) to the original kernel, so the selected
            # documents and their tie ordering match it exactly.
            n_exhaustive += 1
            length = 0
            tau = np.float32(-1.0)
            for d in range(num_docs):
                v = scores[d]
                if weight_mask is not None:
                    v = v * weight_mask[d]
                if length < k:
                    heap_push(values, inds, v, d, length)
                    length += 1
                    if length == k:
                        tau = values[0]
                elif v > tau:
                    values[0] = v
                    inds[0] = d
                    sift_up(values, inds, 0, length)
                    tau = values[0]

        if sorted:
            sorted_inds = np.flip(np.argsort(values))
            topk_scores[i] = values[sorted_inds]
            topk_indices[i] = inds[sorted_inds]

    return topk_scores, topk_indices, n_exhaustive


@njit(parallel=True, cache=True)
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


@njit(parallel=True, cache=True)
def _build_quantized_data(data):
    """Quantize posting scores to uint8 with a single global linear step."""
    mx = np.float32(0.0)
    for j in range(len(data)):
        if data[j] > mx:
            mx = data[j]
    step = np.float64(mx) / 255.0 if mx > 0 else 1.0
    data_q = np.empty(len(data), dtype=np.uint8)
    for j in prange(len(data)):
        q = int(np.float64(data[j]) / step + 0.5)
        if q > 255:
            q = 255
        data_q[j] = q
    return data_q, np.float32(step)


@njit(parallel=True, cache=True)
def _retrieve_internal_jitted_parallel_quantized(
    query_tokens_ids_flat: np.ndarray,
    query_pointers: np.ndarray,
    k: int,
    sorted: bool,
    int_dtype: np.dtype,
    data_q: np.ndarray,
    step: np.float32,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
):
    # Same structure as _retrieve_internal_jitted_parallel, but accumulating
    # uint8-quantized posting scores into a uint16 accumulator: the posting
    # stream shrinks from 8 to 5 bytes per posting and every pass over the
    # scores array moves half the bytes. Scores are approximate (each posting
    # is off by at most step/2), so this kernel is opt-in.
    N = len(query_pointers) - 1
    chunk = _SELECTION_CHUNK_SIZE
    n_chunks = (num_docs + chunk - 1) // chunk

    topk_scores = np.zeros((N, k), dtype=np.float32)
    topk_indices = np.zeros((N, k), dtype=int_dtype)

    for i in prange(N):
        scores = np.zeros(num_docs, dtype=np.uint16)
        for q_ptr in range(query_pointers[i], query_pointers[i + 1]):
            t = query_tokens_ids_flat[q_ptr]
            for j in range(indptr[t], indptr[t + 1]):
                scores[indices[j]] += data_q[j]

        values = np.zeros(k, dtype=np.int64)
        inds = topk_indices[i]
        length = 0
        tau = np.int64(-1)

        if n_chunks > k:
            chunk_maxes = np.empty(n_chunks, dtype=np.uint16)
            for c in range(n_chunks):
                start = c * chunk
                end = min(start + chunk, num_docs)
                m0 = np.uint16(0)
                m1 = np.uint16(0)
                m2 = np.uint16(0)
                m3 = np.uint16(0)
                x = start
                while x + 4 <= end:
                    if scores[x] > m0: m0 = scores[x]
                    if scores[x + 1] > m1: m1 = scores[x + 1]
                    if scores[x + 2] > m2: m2 = scores[x + 2]
                    if scores[x + 3] > m3: m3 = scores[x + 3]
                    x += 4
                while x < end:
                    if scores[x] > m0: m0 = scores[x]
                    x += 1
                if m1 > m0: m0 = m1
                if m2 > m0: m0 = m2
                if m3 > m0: m0 = m3
                chunk_maxes[c] = m0

            # the kth largest chunk max lower-bounds the kth best score (the
            # same argument as the exact kernel)
            bound_values = np.empty(k, dtype=np.int64)
            bound_inds = np.empty(k, dtype=int_dtype)
            bound_length = 0
            for c in range(n_chunks):
                v = np.int64(chunk_maxes[c])
                if bound_length < k:
                    heap_push(bound_values, bound_inds, v, c, bound_length)
                    bound_length += 1
                elif v > bound_values[0]:
                    bound_values[0] = v
                    sift_up(bound_values, bound_inds, 0, bound_length)
            fill_bound = bound_values[0]

            for c in range(n_chunks):
                if length >= k:
                    if np.int64(chunk_maxes[c]) <= tau:
                        continue
                elif np.int64(chunk_maxes[c]) < fill_bound:
                    continue
                start = c * chunk
                end = min(start + chunk, num_docs)
                for d in range(start, end):
                    v = np.int64(scores[d])
                    if length < k:
                        if v >= fill_bound:
                            heap_push(values, inds, v, d, length)
                            length += 1
                            if length == k:
                                tau = values[0]
                    elif v > tau:
                        values[0] = v
                        inds[0] = d
                        sift_up(values, inds, 0, length)
                        tau = values[0]
        else:
            for d in range(num_docs):
                v = np.int64(scores[d])
                if length < k:
                    heap_push(values, inds, v, d, length)
                    length += 1
                    if length == k:
                        tau = values[0]
                elif v > tau:
                    values[0] = v
                    inds[0] = d
                    sift_up(values, inds, 0, length)
                    tau = values[0]

        if sorted:
            sorted_inds = np.flip(np.argsort(values))
            for x in range(k):
                topk_scores[i, x] = np.float32(values[sorted_inds[x]]) * step
            reordered = inds[sorted_inds]
            for x in range(k):
                inds[x] = reordered[x]
        else:
            for x in range(k):
                topk_scores[i, x] = np.float32(values[x]) * step

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
    exact_ties=False,
    quantize=False,
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

    if quantize:
        if nonoccurrence_array is not None:
            raise ValueError(
                "quantize=True is not supported with the 'bm25l'/'bm25+' methods."
            )
        if weight_mask is not None:
            raise ValueError("quantize=True is not supported with weight_mask.")
        if exact_ties:
            raise ValueError(
                "quantize=True computes approximate scores and cannot honor exact_ties=True."
            )
        # the quantized postings are derived from the float index once and
        # cached on the scores dict (save/load only persists the named arrays)
        if "data_q" not in scores:
            scores["data_q"], scores["q_step"] = _build_quantized_data(scores["data"])
        retrieved_scores, retrieved_indices = _retrieve_internal_jitted_parallel_quantized(
            query_tokens_ids_flat=query_tokens_ids_flat,
            query_pointers=query_pointers,
            k=k,
            sorted=sorted,
            int_dtype=np.dtype(int_dtype),
            data_q=scores["data_q"],
            step=scores["q_step"],
            indptr=scores["indptr"],
            indices=scores["indices"],
            num_docs=scores["num_docs"],
        )
    elif nonoccurrence_array is None:
        kernel_kwargs = dict(
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
        # Both kernels below may be used depending on the tie rate, so make
        # sure both are compiled on the first call (e.g. a warmup): otherwise
        # the exhaustive kernel would be JIT-compiled in the middle of the
        # first large retrieval. The warmup runs a single query and is keyed
        # on the argument types that determine the compiled specialization.
        warm_key = (
            str(np.dtype(dtype)),
            str(np.dtype(int_dtype)),
            str(scores["indptr"].dtype),
            str(scores["indices"].dtype),
            bool(sorted),
            weight_mask is None,
        )
        if warm_key not in _EXHAUSTIVE_KERNEL_WARMED and len(query_pointers) > 1:
            _EXHAUSTIVE_KERNEL_WARMED.add(warm_key)
            _retrieve_internal_jitted_parallel_nonoccurrence(
                query_pointers=query_pointers[:2],
                query_tokens_ids_flat=query_tokens_ids_flat,
                nonoccurrence_array=None,
                **kernel_kwargs,
            )
        # The pruned selection is exact but reverts to the exhaustive scan for
        # queries whose top-k contains tied scores; if a probe of the first
        # queries shows that ties dominate (as is common with large k), skip
        # the pruning attempt for the remaining queries.
        n_queries = len(query_pointers) - 1
        n_chunks = (scores["num_docs"] + _SELECTION_CHUNK_SIZE - 1) // _SELECTION_CHUNK_SIZE
        n_probe = min(32, n_queries) if n_chunks > k else 0
        if n_probe > 0:
            retrieved_scores, retrieved_indices, n_exhaustive = _retrieve_internal_jitted_parallel(
                query_pointers=query_pointers[: n_probe + 1],
                query_tokens_ids_flat=query_tokens_ids_flat,
                exact_ties=exact_ties,
                **kernel_kwargs,
            )
        else:
            retrieved_scores = retrieved_indices = None
            n_exhaustive = 1  # pruning cannot engage: use the exhaustive kernel
        if n_queries > n_probe:
            rest_pointers = query_pointers[n_probe:] - query_pointers[n_probe]
            rest_tokens = query_tokens_ids_flat[query_pointers[n_probe]:]
            if 2 * n_exhaustive < max(n_probe, 1):
                rest_scores, rest_indices, _ = _retrieve_internal_jitted_parallel(
                    query_pointers=rest_pointers,
                    query_tokens_ids_flat=rest_tokens,
                    exact_ties=exact_ties,
                    **kernel_kwargs,
                )
            else:
                # ties dominate, so the pruning attempts would be wasted work:
                # run the exhaustive kernel directly (identical results)
                rest_scores, rest_indices = _retrieve_internal_jitted_parallel_nonoccurrence(
                    query_pointers=rest_pointers,
                    query_tokens_ids_flat=rest_tokens,
                    nonoccurrence_array=None,
                    **kernel_kwargs,
                )
            if retrieved_scores is None:
                retrieved_scores, retrieved_indices = rest_scores, rest_indices
            else:
                retrieved_scores = np.concatenate((retrieved_scores, rest_scores))
                retrieved_indices = np.concatenate((retrieved_indices, rest_indices))
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
