import logging
import os
from typing import Any, List

import numpy as np

from .. import utils
from .selection import _require_cupy, _topk_cupy_gpu


_CANDIDATE_SCORE_KERNEL = None
_CANDIDATE_SCORE_COUNT_KERNEL = None
_CANDIDATE_MARK_KERNEL = None
_CANDIDATE_STAMP_KERNEL = None
_CANDIDATE_DROPPED_SCORE_KERNEL = None
_WARNED_IGNORED_CHUNKSIZE = False
_WARNED_IGNORED_N_THREADS = False


def _get_candidate_score_kernel():
    global _CANDIDATE_SCORE_KERNEL
    if _CANDIDATE_SCORE_KERNEL is not None:
        return _CANDIDATE_SCORE_KERNEL

    cupy = _require_cupy()
    _CANDIDATE_SCORE_KERNEL = cupy.RawKernel(
        r'''
extern "C" __global__ void score_candidates_kernel(
    const int* candidates,
    const int n_candidates,
    const int* query_terms,
    const int q_len,
    const float* data,
    const int* indices,
    const int* indptr,
    float* out_scores) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_candidates) return;

  int doc = candidates[i];
  float score = 0.0f;
  for (int t = 0; t < q_len; ++t) {
    int term = query_terms[t];
    int lo = indptr[term];
    int hi = indptr[term + 1];
    while (lo < hi) {
      int mid = lo + ((hi - lo) >> 1);
      int value = indices[mid];
      if (value < doc) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int end = indptr[term + 1];
    if (lo < end && indices[lo] == doc) {
      score += data[lo];
    }
  }
  out_scores[i] = score;
}
''',
        "score_candidates_kernel",
    )
    return _CANDIDATE_SCORE_KERNEL


def _get_candidate_score_count_kernel():
    global _CANDIDATE_SCORE_COUNT_KERNEL
    if _CANDIDATE_SCORE_COUNT_KERNEL is not None:
        return _CANDIDATE_SCORE_COUNT_KERNEL

    cupy = _require_cupy()
    _CANDIDATE_SCORE_COUNT_KERNEL = cupy.RawKernel(
        r'''
extern "C" __global__ void score_candidates_count_kernel(
    const int* candidates,
    const int total_slots,
    const int* count,
    const int* query_terms,
    const int q_len,
    const float* data,
    const int* indices,
    const int* indptr,
    float* out_scores) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= total_slots) return;

  int n_candidates = count[0];
  if (i >= n_candidates) {
    out_scores[i] = -3.402823466e+38F;
    return;
  }

  int doc = candidates[i];
  float score = 0.0f;
  for (int t = 0; t < q_len; ++t) {
    int term = query_terms[t];
    int lo = indptr[term];
    int hi = indptr[term + 1];
    while (lo < hi) {
      int mid = lo + ((hi - lo) >> 1);
      int value = indices[mid];
      if (value < doc) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int end = indptr[term + 1];
    if (lo < end && indices[lo] == doc) {
      score += data[lo];
    }
  }
  out_scores[i] = score;
}
''',
        "score_candidates_count_kernel",
    )
    return _CANDIDATE_SCORE_COUNT_KERNEL


def _get_candidate_mark_kernel():
    global _CANDIDATE_MARK_KERNEL
    if _CANDIDATE_MARK_KERNEL is not None:
        return _CANDIDATE_MARK_KERNEL

    cupy = _require_cupy()
    _CANDIDATE_MARK_KERNEL = cupy.RawKernel(
        r'''
extern "C" __global__ void build_candidates_mark_kernel(
    const int* term_starts,
    const int* term_offsets,
    const int n_terms,
    const int total_postings,
    const int* indices,
    int* marks,
    int* candidates,
    int* count) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= total_postings) return;

  int lo = 0;
  int hi = n_terms;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (term_offsets[mid + 1] <= pos) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  int term_idx = lo;
  int src = term_starts[term_idx] + (pos - term_offsets[term_idx]);
  int doc = indices[src];
  if (atomicCAS(&marks[doc], 0, 1) == 0) {
    int out = atomicAdd(count, 1);
    candidates[out] = doc;
  }
}
''',
        "build_candidates_mark_kernel",
    )
    return _CANDIDATE_MARK_KERNEL


def _get_candidate_stamp_kernel():
    global _CANDIDATE_STAMP_KERNEL
    if _CANDIDATE_STAMP_KERNEL is not None:
        return _CANDIDATE_STAMP_KERNEL

    cupy = _require_cupy()
    _CANDIDATE_STAMP_KERNEL = cupy.RawKernel(
        r'''
extern "C" __global__ void build_candidates_stamp_kernel(
    const int* term_starts,
    const int* term_offsets,
    const int n_terms,
    const int total_postings,
    const int* indices,
    int* marks,
    const int stamp,
    int* candidates,
    int* count) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= total_postings) return;

  int lo = 0;
  int hi = n_terms;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (term_offsets[mid + 1] <= pos) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  int term_idx = lo;
  int src = term_starts[term_idx] + (pos - term_offsets[term_idx]);
  int doc = indices[src];
  int old = atomicExch(&marks[doc], stamp);
  if (old != stamp) {
    int out = atomicAdd(count, 1);
    candidates[out] = doc;
  }
}
''',
        "build_candidates_stamp_kernel",
    )
    return _CANDIDATE_STAMP_KERNEL


def _get_candidate_dropped_score_kernel():
    global _CANDIDATE_DROPPED_SCORE_KERNEL
    if _CANDIDATE_DROPPED_SCORE_KERNEL is not None:
        return _CANDIDATE_DROPPED_SCORE_KERNEL

    cupy = _require_cupy()
    _CANDIDATE_DROPPED_SCORE_KERNEL = cupy.RawKernel(
        r'''
extern "C" __global__ void add_dropped_scores_kernel(
    const int* candidate_docs,
    const int nnz,
    const int* row_ids,
    const int* drop_terms,
    const int* drop_offsets,
    const float* data,
    const int* indices,
    const int* indptr,
    float* candidate_scores) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= nnz) return;

  int row = row_ids[i];
  int doc = candidate_docs[i];
  float score = candidate_scores[i];
  for (int p = drop_offsets[row]; p < drop_offsets[row + 1]; ++p) {
    int term = drop_terms[p];
    int lo = indptr[term];
    int hi = indptr[term + 1];
    while (lo < hi) {
      int mid = lo + ((hi - lo) >> 1);
      int value = indices[mid];
      if (value < doc) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int end = indptr[term + 1];
    if (lo < end && indices[lo] == doc) {
      score += data[lo];
    }
  }
  candidate_scores[i] = score;
}
''',
        "add_dropped_scores_kernel",
    )
    return _CANDIDATE_DROPPED_SCORE_KERNEL


def _get_cupy_index_cache(scores, dtype, int_dtype):
    cupy = _require_cupy()
    cache = scores.get("_cupy_cache")
    cache_key = (str(np.dtype(dtype)), str(np.dtype(int_dtype)))
    if cache is not None and cache.get("key") == cache_key:
        return cache

    indptr_cpu = np.asarray(scores["indptr"], dtype=int_dtype)
    cache = {
        "key": cache_key,
        "data": cupy.asarray(scores["data"], dtype=dtype),
        "indices": cupy.asarray(scores["indices"], dtype=int_dtype),
        "indptr_gpu": cupy.asarray(indptr_cpu, dtype=int_dtype),
        "indptr_cpu": indptr_cpu,
    }
    scores["_cupy_cache"] = cache
    return cache


def _get_term_max_cache(scores, dtype):
    cache = scores.get("_term_max_cache")
    cache_key = str(np.dtype(dtype))
    if cache is not None and cache.get("key") == cache_key:
        return cache["term_max"]

    data = np.asarray(scores["data"], dtype=dtype)
    indptr = np.asarray(scores["indptr"])
    term_max = np.zeros(len(indptr) - 1, dtype=dtype)
    for term_id in range(len(term_max)):
        start = int(indptr[term_id])
        end = int(indptr[term_id + 1])
        if end > start:
            term_max[term_id] = data[start:end].max()
    scores["_term_max_cache"] = {"key": cache_key, "term_max": term_max}
    return term_max


def _get_cupy_spmm_cache(scores, dtype, int_dtype):
    cache = scores.get("_cupy_spmm_cache")
    cache_key = (str(np.dtype(dtype)), str(np.dtype(int_dtype)))
    if cache is not None and cache.get("key") == cache_key:
        return cache

    cupy = _require_cupy()
    import cupyx.scipy.sparse as cupy_sparse

    index_cache = _get_cupy_index_cache(scores, dtype=dtype, int_dtype=int_dtype)
    num_docs = int(scores["num_docs"])
    n_terms = len(index_cache["indptr_cpu"]) - 1
    score_matrix = cupy_sparse.csc_matrix(
        (
            index_cache["data"],
            index_cache["indices"],
            index_cache["indptr_gpu"],
        ),
        shape=(num_docs, n_terms),
    )
    cache = {
        "key": cache_key,
        "score_matrix_t": score_matrix.T.tocsr(),
        "n_terms": n_terms,
        "empty_int": cupy.empty(0, dtype=int_dtype),
        "empty_float": cupy.empty(0, dtype=dtype),
    }
    scores["_cupy_spmm_cache"] = cache
    return cache


def _unique_sorted_cupy(values):
    cupy = _require_cupy()

    values = cupy.sort(values)
    if values.size == 0:
        return values

    keep = cupy.empty(values.shape, dtype=cupy.bool_)
    keep[0] = True
    keep[1:] = values[1:] != values[:-1]
    return values[keep]


def _candidate_terms_from_query(query_terms, indptr_cpu, df_threshold: int):
    dfs = indptr_cpu[query_terms + 1] - indptr_cpu[query_terms]
    kept_terms = query_terms[dfs <= df_threshold]
    dropped_terms = query_terms[dfs > df_threshold]
    return np.unique(kept_terms), dropped_terms


def _resolve_candidate_certification(
    candidate_positions,
    topk_scores,
    outside_bounds,
):
    if not candidate_positions:
        return []

    cupy = _require_cupy()
    kth_scores = cupy.asnumpy(cupy.stack([scores[-1] for scores in topk_scores]))
    return [
        query_pos
        for query_pos, kth_score, outside_bound in zip(
            candidate_positions, kth_scores, outside_bounds
        )
        if kth_score <= outside_bound
    ]


def _build_candidates_sort_cupy(candidate_terms, indptr_cpu, indices_gpu):
    cupy = _require_cupy()
    candidate_parts = [
        indices_gpu[int(indptr_cpu[term_id]) : int(indptr_cpu[term_id + 1])]
        for term_id in candidate_terms
    ]
    return _unique_sorted_cupy(cupy.concatenate(candidate_parts))


def _build_candidates_atomic_cupy(
    candidate_terms,
    indptr_cpu,
    indices_gpu,
    marks_gpu,
    count_gpu,
):
    cupy = _require_cupy()
    starts = indptr_cpu[candidate_terms].astype(np.int32, copy=False)
    ends = indptr_cpu[candidate_terms + 1].astype(np.int32, copy=False)
    posting_lengths = (ends - starts).astype(np.int32, copy=False)
    total_postings = int(posting_lengths.sum())
    if total_postings == 0:
        return cupy.empty(0, dtype=indices_gpu.dtype)

    if total_postings > np.iinfo(np.int32).max:
        return _build_candidates_sort_cupy(candidate_terms, indptr_cpu, indices_gpu)

    offsets = np.empty(len(posting_lengths) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(posting_lengths, dtype=np.int32, out=offsets[1:])

    candidates = cupy.empty(total_postings, dtype=indices_gpu.dtype)
    count_gpu.fill(0)
    block_size = 256
    grid_size = (total_postings + block_size - 1) // block_size
    _get_candidate_mark_kernel()(
        (grid_size,),
        (block_size,),
        (
            cupy.asarray(starts),
            cupy.asarray(offsets),
            np.int32(len(posting_lengths)),
            np.int32(total_postings),
            indices_gpu,
            marks_gpu,
            candidates,
            count_gpu,
        ),
    )
    n_candidates = int(count_gpu.get()[0])
    candidates = candidates[:n_candidates]
    if n_candidates:
        marks_gpu[candidates] = 0
    return candidates


def _build_candidates_stamp_cupy(
    candidate_terms,
    indptr_cpu,
    indices_gpu,
    marks_gpu,
    count_gpu,
    stamp: int,
):
    cupy = _require_cupy()
    starts = indptr_cpu[candidate_terms].astype(np.int32, copy=False)
    ends = indptr_cpu[candidate_terms + 1].astype(np.int32, copy=False)
    posting_lengths = (ends - starts).astype(np.int32, copy=False)
    total_postings = int(posting_lengths.sum())
    if total_postings == 0:
        return cupy.empty(0, dtype=indices_gpu.dtype)

    if total_postings > np.iinfo(np.int32).max:
        return _build_candidates_sort_cupy(candidate_terms, indptr_cpu, indices_gpu)

    offsets = np.empty(len(posting_lengths) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(posting_lengths, dtype=np.int32, out=offsets[1:])

    candidates = cupy.empty(total_postings, dtype=indices_gpu.dtype)
    count_gpu.fill(0)
    block_size = 256
    grid_size = (total_postings + block_size - 1) // block_size
    _get_candidate_stamp_kernel()(
        (grid_size,),
        (block_size,),
        (
            cupy.asarray(starts),
            cupy.asarray(offsets),
            np.int32(len(posting_lengths)),
            np.int32(total_postings),
            indices_gpu,
            marks_gpu,
            np.int32(stamp),
            candidates,
            count_gpu,
        ),
    )
    n_candidates = int(count_gpu.get()[0])
    return candidates[:n_candidates]


def _build_candidates_stamp_slots_cupy(
    candidate_terms,
    indptr_cpu,
    indices_gpu,
    marks_gpu,
    count_gpu,
    stamp: int,
):
    cupy = _require_cupy()
    starts = indptr_cpu[candidate_terms].astype(np.int32, copy=False)
    ends = indptr_cpu[candidate_terms + 1].astype(np.int32, copy=False)
    posting_lengths = (ends - starts).astype(np.int32, copy=False)
    total_postings = int(posting_lengths.sum())
    if total_postings == 0:
        return cupy.empty(0, dtype=indices_gpu.dtype), 0, count_gpu

    if total_postings > np.iinfo(np.int32).max:
        candidates = _build_candidates_sort_cupy(candidate_terms, indptr_cpu, indices_gpu)
        count_gpu[0] = candidates.size
        return candidates, int(candidates.size), count_gpu

    offsets = np.empty(len(posting_lengths) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(posting_lengths, dtype=np.int32, out=offsets[1:])

    candidates = cupy.empty(total_postings, dtype=indices_gpu.dtype)
    count_gpu.fill(0)
    block_size = 256
    grid_size = (total_postings + block_size - 1) // block_size
    _get_candidate_stamp_kernel()(
        (grid_size,),
        (block_size,),
        (
            cupy.asarray(starts),
            cupy.asarray(offsets),
            np.int32(len(posting_lengths)),
            np.int32(total_postings),
            indices_gpu,
            marks_gpu,
            np.int32(stamp),
            candidates,
            count_gpu,
        ),
    )
    return candidates, total_postings, count_gpu


def _build_query_low_df_csr_cupy(
    query_batch,
    indptr_cpu,
    df_threshold,
    n_terms,
    dtype,
    int_dtype,
):
    cupy = _require_cupy()
    import cupyx.scipy.sparse as cupy_sparse

    q_indptr = [0]
    q_indices = []
    q_data = []
    drop_offsets = [0]
    drop_terms = []
    for query_terms in query_batch:
        query_terms = np.asarray(query_terms, dtype=int_dtype)
        if query_terms.size:
            dfs = indptr_cpu[query_terms + 1] - indptr_cpu[query_terms]
            low_terms = query_terms[dfs <= df_threshold]
            dropped = query_terms[dfs > df_threshold]
            if low_terms.size:
                q_indices.extend(low_terms.tolist())
                q_data.extend([1.0] * int(low_terms.size))
            drop_terms.extend(dropped.tolist())
        q_indptr.append(len(q_indices))
        drop_offsets.append(len(drop_terms))

    query_matrix = cupy_sparse.csr_matrix(
        (
            cupy.asarray(q_data, dtype=dtype),
            cupy.asarray(q_indices, dtype=int_dtype),
            cupy.asarray(q_indptr, dtype=int_dtype),
        ),
        shape=(len(query_batch), n_terms),
    )
    return (
        query_matrix,
        cupy.asarray(drop_terms, dtype=int_dtype),
        cupy.asarray(drop_offsets, dtype=int_dtype),
    )


def _add_dropped_scores_to_csr_cupy(
    candidates_csr,
    drop_terms,
    drop_offsets,
    scores_cache,
):
    cupy = _require_cupy()
    nnz = int(candidates_csr.nnz)
    if nnz == 0 or int(drop_terms.size) == 0:
        return

    row_counts = candidates_csr.indptr[1:] - candidates_csr.indptr[:-1]
    row_ids = cupy.repeat(cupy.arange(candidates_csr.shape[0], dtype=cupy.int32), row_counts)
    block_size = 256
    grid_size = (nnz + block_size - 1) // block_size
    _get_candidate_dropped_score_kernel()(
        (grid_size,),
        (block_size,),
        (
            candidates_csr.indices,
            np.int32(nnz),
            row_ids,
            drop_terms,
            drop_offsets,
            scores_cache["data"],
            scores_cache["indices"],
            scores_cache["indptr_gpu"],
            candidates_csr.data,
        ),
    )


def _topk_from_csr_rows_cupy(candidates_csr, k, dtype, int_dtype, indptr_cpu=None):
    cupy = _require_cupy()
    docs_rows = []
    score_rows = []
    fallback_rows = []
    if indptr_cpu is None:
        indptr_cpu = cupy.asnumpy(candidates_csr.indptr)
    for row in range(candidates_csr.shape[0]):
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        row_scores = candidates_csr.data[start:end]
        if int(row_scores.size) < k:
            fallback_rows.append(row)
            docs_rows.append(None)
            score_rows.append(None)
            continue
        row_docs = candidates_csr.indices[start:end]
        order = cupy.argsort(row_scores)[-k:][::-1]
        docs_rows.append(row_docs[order].astype(int_dtype, copy=False))
        score_rows.append(row_scores[order].astype(dtype, copy=False))
    return docs_rows, score_rows, fallback_rows


def _retrieve_sparse_batch_cupy(
    query_tokens_ids,
    scores,
    k: int,
    dtype,
    int_dtype,
    max_postings: int = 50_000_000,
):
    """Retrieve a sparse batch exactly when all rows have at least k non-zero docs.

    BM25 score entries are non-negative for the methods that use this fast path.
    Therefore, if a query has at least k scored documents, its dense top-k is
    contained in the non-zero candidate set and we can avoid scanning all docs.
    """
    cupy = _require_cupy()
    dtype = np.dtype(dtype)
    int_dtype = np.dtype(int_dtype)
    num_queries = len(query_tokens_ids)
    num_docs = int(scores["num_docs"])
    if num_queries == 0 or k == 0:
        return (
            np.empty((num_queries, k), dtype=int_dtype),
            np.empty((num_queries, k), dtype=dtype),
        )

    query_lengths = np.asarray([len(q) for q in query_tokens_ids], dtype=np.int64)
    if np.any(query_lengths == 0):
        return None

    cache = _get_cupy_index_cache(scores, dtype=dtype, int_dtype=int_dtype)
    data_gpu = cache["data"]
    indices_gpu = cache["indices"]
    indptr_cpu = cache["indptr_cpu"]

    term_ids = np.concatenate(
        [np.asarray(q, dtype=int_dtype) for q in query_tokens_ids]
    )
    starts = indptr_cpu[term_ids].astype(np.int64, copy=False)
    ends = indptr_cpu[term_ids + 1].astype(np.int64, copy=False)
    posting_lengths = ends - starts
    total_postings = int(posting_lengths.sum())
    if total_postings == 0 or total_postings > max_postings:
        return None

    offsets = np.empty(len(posting_lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(posting_lengths, dtype=np.int64, out=offsets[1:])
    term_rows = np.repeat(np.arange(num_queries, dtype=np.int32), query_lengths)

    offsets_gpu = cupy.asarray(offsets)
    starts_gpu = cupy.asarray(starts)
    rows_gpu = cupy.asarray(term_rows)

    pos = cupy.arange(total_postings, dtype=cupy.int64)
    term_ix = cupy.searchsorted(offsets_gpu[1:], pos, side="right")
    src = starts_gpu[term_ix] + (pos - offsets_gpu[term_ix])
    rows = rows_gpu[term_ix].astype(cupy.int64)
    docs = indices_gpu[src].astype(cupy.int64)
    vals = data_gpu[src]

    keys = rows * num_docs + docs
    order = cupy.argsort(keys)
    keys = keys[order]
    vals = vals[order]

    is_new = cupy.empty(keys.shape, dtype=cupy.bool_)
    is_new[0] = True
    is_new[1:] = keys[1:] != keys[:-1]
    group = cupy.cumsum(is_new, dtype=cupy.int64) - 1
    n_groups = int(group[-1].get()) + 1

    summed_scores = cupy.zeros(n_groups, dtype=dtype)
    cupy.add.at(summed_scores, group, vals)
    unique_keys = keys[is_new]
    unique_rows = (unique_keys // num_docs).astype(cupy.int32)
    unique_docs = (unique_keys - unique_rows.astype(cupy.int64) * num_docs).astype(
        cupy.int32
    )

    counts = cupy.bincount(unique_rows, minlength=num_queries)
    if bool(cupy.any(counts < k).get()):
        return None

    lex_keys = cupy.stack(
        (
            unique_docs.astype(cupy.float64),
            (-summed_scores).astype(cupy.float64),
            unique_rows.astype(cupy.float64),
        )
    )
    order = cupy.lexsort(lex_keys)
    rows_sorted = unique_rows[order]
    docs_sorted = unique_docs[order]
    scores_sorted = summed_scores[order]

    sorted_counts = cupy.bincount(rows_sorted, minlength=num_queries)
    row_starts = cupy.cumsum(
        cupy.concatenate(
            [cupy.asarray([0], dtype=sorted_counts.dtype), sorted_counts[:-1]]
        )
    )
    ranks = cupy.arange(rows_sorted.size, dtype=cupy.int64) - cupy.repeat(
        row_starts, sorted_counts
    ).astype(cupy.int64)
    keep = ranks < k

    retrieved_indices = cupy.empty((num_queries, k), dtype=int_dtype)
    retrieved_scores = cupy.empty((num_queries, k), dtype=dtype)
    flat_positions = rows_sorted[keep].astype(cupy.int64) * k + ranks[keep]
    retrieved_indices.ravel()[flat_positions] = docs_sorted[keep].astype(int_dtype)
    retrieved_scores.ravel()[flat_positions] = scores_sorted[keep].astype(dtype)

    return cupy.asnumpy(retrieved_indices), cupy.asnumpy(retrieved_scores)


def _retrieve_spmm_batch_cupy(
    query_tokens_ids,
    scores,
    k: int,
    dtype,
    int_dtype,
    df_threshold: int,
    batch_size: int = 160,
):
    cupy = _require_cupy()
    dtype = np.dtype(dtype)
    int_dtype = np.dtype(int_dtype)
    if dtype != np.dtype("float32") or int_dtype != np.dtype("int32"):
        return None

    num_queries = len(query_tokens_ids)
    if num_queries == 0:
        return (
            np.empty((0, k), dtype=int_dtype),
            np.empty((0, k), dtype=dtype),
        )

    index_cache = _get_cupy_index_cache(scores, dtype=dtype, int_dtype=int_dtype)
    spmm_cache = _get_cupy_spmm_cache(scores, dtype=dtype, int_dtype=int_dtype)
    indptr_cpu = index_cache["indptr_cpu"]
    term_max = _get_term_max_cache(scores, dtype=dtype)

    retrieved_docs = [None] * num_queries
    retrieved_scores = [None] * num_queries
    fallback_positions = []
    cert_candidate_positions = []
    cert_topk_scores = []
    cert_outside_bounds = []

    batch_size = max(int(batch_size), 1)
    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        query_batch = [
            np.asarray(query_tokens_ids[pos], dtype=int_dtype)
            for pos in range(start, end)
        ]
        if any(len(query_terms) == 0 for query_terms in query_batch):
            for local_pos, query_terms in enumerate(query_batch):
                if len(query_terms) == 0:
                    fallback_positions.append(start + local_pos)

        query_matrix, drop_terms_gpu, drop_offsets_gpu = _build_query_low_df_csr_cupy(
            query_batch,
            indptr_cpu=indptr_cpu,
            df_threshold=df_threshold,
            n_terms=spmm_cache["n_terms"],
            dtype=dtype,
            int_dtype=int_dtype,
        )
        if int(query_matrix.nnz) == 0:
            fallback_positions.extend(range(start, end))
            continue

        candidate_scores_csr = query_matrix @ spmm_cache["score_matrix_t"]
        _add_dropped_scores_to_csr_cupy(
            candidate_scores_csr,
            drop_terms=drop_terms_gpu,
            drop_offsets=drop_offsets_gpu,
            scores_cache=index_cache,
        )
        candidate_indptr_cpu = cupy.asnumpy(candidate_scores_csr.indptr)
        docs_rows, score_rows, local_fallback_rows = _topk_from_csr_rows_cupy(
            candidate_scores_csr,
            k=k,
            dtype=dtype,
            int_dtype=int_dtype,
            indptr_cpu=candidate_indptr_cpu,
        )
        fallback_positions.extend(start + row for row in local_fallback_rows)

        for local_pos, (docs_row, scores_row) in enumerate(zip(docs_rows, score_rows)):
            query_pos = start + local_pos
            if docs_row is None:
                continue
            query_terms = query_batch[local_pos]
            dfs = indptr_cpu[query_terms + 1] - indptr_cpu[query_terms]
            dropped_terms = query_terms[dfs > df_threshold]
            outside_bound = (
                float(term_max[dropped_terms].sum()) if len(dropped_terms) else 0.0
            )
            retrieved_docs[query_pos] = docs_row
            retrieved_scores[query_pos] = scores_row
            cert_candidate_positions.append(query_pos)
            cert_topk_scores.append(scores_row)
            cert_outside_bounds.append(outside_bound)

    fallback_positions.extend(
        _resolve_candidate_certification(
            candidate_positions=cert_candidate_positions,
            topk_scores=cert_topk_scores,
            outside_bounds=cert_outside_bounds,
        )
    )
    fallback_positions = sorted(set(fallback_positions))
    for query_pos in fallback_positions:
        retrieved_docs[query_pos] = None
        retrieved_scores[query_pos] = None

    if fallback_positions:
        fallback_docs, fallback_scores = _retrieve_dense_batch_cupy(
            [query_tokens_ids[pos] for pos in fallback_positions],
            scores,
            k=k,
            sorted=True,
            dtype=dtype,
            int_dtype=int_dtype,
        )
        for fallback_idx, query_pos in enumerate(fallback_positions):
            retrieved_docs[query_pos] = cupy.asarray(
                fallback_docs[fallback_idx], dtype=int_dtype
            )
            retrieved_scores[query_pos] = cupy.asarray(
                fallback_scores[fallback_idx], dtype=dtype
            )

    return (
        cupy.asnumpy(cupy.stack(retrieved_docs)),
        cupy.asnumpy(cupy.stack(retrieved_scores)),
    )


def _retrieve_dense_batch_cupy(
    query_tokens_ids,
    scores,
    k: int,
    sorted: bool,
    dtype,
    int_dtype,
    nonoccurrence_array=None,
    weight_mask=None,
):
    cupy = _require_cupy()
    dtype = np.dtype(dtype)
    int_dtype = np.dtype(int_dtype)
    num_queries = len(query_tokens_ids)
    if num_queries == 0:
        return (
            np.empty((0, k), dtype=int_dtype),
            np.empty((0, k), dtype=dtype),
        )

    cache = _get_cupy_index_cache(scores, dtype=dtype, int_dtype=int_dtype)
    data_gpu = cache["data"]
    indices_gpu = cache["indices"]
    indptr = cache["indptr_cpu"]

    if nonoccurrence_array is not None:
        nonoccurrence_array_gpu = cupy.asarray(nonoccurrence_array, dtype=dtype)
    else:
        nonoccurrence_array_gpu = None

    if weight_mask is not None:
        weight_mask_gpu = cupy.asarray(weight_mask)
    else:
        weight_mask_gpu = None

    retrieved_scores_gpu = []
    retrieved_indices_gpu = []
    for query_tokens_ids_single in query_tokens_ids:
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

        topk_scores_sing, topk_indices_sing = _topk_cupy_gpu(
            scores_single, k=k, sorted=sorted
        )
        retrieved_scores_gpu.append(topk_scores_sing.astype(dtype, copy=False))
        retrieved_indices_gpu.append(topk_indices_sing.astype(int_dtype, copy=False))

    return (
        cupy.asnumpy(cupy.stack(retrieved_indices_gpu)),
        cupy.asnumpy(cupy.stack(retrieved_scores_gpu)),
    )


def _retrieve_candidate_batch_cupy(
    query_tokens_ids,
    scores,
    k: int,
    dtype,
    int_dtype,
    df_threshold: int,
):
    cupy = _require_cupy()
    dtype = np.dtype(dtype)
    int_dtype = np.dtype(int_dtype)
    if dtype != np.dtype("float32") or int_dtype != np.dtype("int32"):
        return None

    num_queries = len(query_tokens_ids)
    if num_queries == 0:
        return (
            np.empty((0, k), dtype=int_dtype),
            np.empty((0, k), dtype=dtype),
        )

    union_mode = os.environ.get("BM25S_CUPY_CANDIDATE_UNION", "spmm")
    if union_mode == "spmm":
        return _retrieve_spmm_batch_cupy(
            query_tokens_ids=query_tokens_ids,
            scores=scores,
            k=k,
            dtype=dtype,
            int_dtype=int_dtype,
            df_threshold=df_threshold,
            batch_size=int(os.environ.get("BM25S_CUPY_SPMM_BATCH_SIZE", "256")),
        )

    cache = _get_cupy_index_cache(scores, dtype=dtype, int_dtype=int_dtype)
    data_gpu = cache["data"]
    indices_gpu = cache["indices"]
    indptr_gpu = cache["indptr_gpu"]
    indptr_cpu = cache["indptr_cpu"]
    term_max = _get_term_max_cache(scores, dtype=dtype)
    kernel = _get_candidate_score_kernel()
    count_kernel = _get_candidate_score_count_kernel()
    if union_mode in {"atomic", "stamp", "stamp_slots"}:
        candidate_marks_gpu = cupy.zeros(int(scores["num_docs"]), dtype=cupy.int32)
        candidate_count_gpu = cupy.zeros(1, dtype=cupy.int32)
    else:
        candidate_marks_gpu = None
        candidate_count_gpu = None

    retrieved_docs = [None] * num_queries
    retrieved_scores = [None] * num_queries
    fallback_positions = []
    cert_candidate_positions = []
    cert_topk_scores = []
    cert_outside_bounds = []

    for query_pos, query_tokens_ids_single in enumerate(query_tokens_ids):
        query_tokens_ids_single = np.asarray(query_tokens_ids_single, dtype=int_dtype)
        if len(query_tokens_ids_single) == 0:
            fallback_positions.append(query_pos)
            continue

        candidate_terms, dropped_terms = _candidate_terms_from_query(
            query_terms=query_tokens_ids_single,
            indptr_cpu=indptr_cpu,
            df_threshold=df_threshold,
        )
        if len(candidate_terms) == 0:
            fallback_positions.append(query_pos)
            continue

        if union_mode == "stamp_slots":
            candidates, n_score_slots, count_gpu = _build_candidates_stamp_slots_cupy(
                candidate_terms=candidate_terms,
                indptr_cpu=indptr_cpu,
                indices_gpu=indices_gpu,
                marks_gpu=candidate_marks_gpu,
                count_gpu=candidate_count_gpu,
                stamp=query_pos + 1,
            )
            n_candidates = n_score_slots
        elif union_mode == "stamp":
            candidates = _build_candidates_stamp_cupy(
                candidate_terms=candidate_terms,
                indptr_cpu=indptr_cpu,
                indices_gpu=indices_gpu,
                marks_gpu=candidate_marks_gpu,
                count_gpu=candidate_count_gpu,
                stamp=query_pos + 1,
            )
            n_score_slots = int(candidates.size)
            count_gpu = None
        elif union_mode == "atomic":
            candidates = _build_candidates_atomic_cupy(
                candidate_terms=candidate_terms,
                indptr_cpu=indptr_cpu,
                indices_gpu=indices_gpu,
                marks_gpu=candidate_marks_gpu,
                count_gpu=candidate_count_gpu,
            )
            n_score_slots = int(candidates.size)
            count_gpu = None
        else:
            candidates = _build_candidates_sort_cupy(
                candidate_terms=candidate_terms,
                indptr_cpu=indptr_cpu,
                indices_gpu=indices_gpu,
            )
            n_score_slots = int(candidates.size)
            count_gpu = None

        n_candidates = int(candidates.size)
        if n_candidates < k:
            fallback_positions.append(query_pos)
            continue

        block_size = 256
        grid_size = (n_score_slots + block_size - 1) // block_size
        candidate_scores = cupy.empty(n_score_slots, dtype=dtype)
        query_terms_gpu = cupy.asarray(query_tokens_ids_single, dtype=int_dtype)
        if union_mode == "stamp_slots":
            count_kernel(
                (grid_size,),
                (block_size,),
                (
                    candidates,
                    np.int32(n_score_slots),
                    count_gpu,
                    query_terms_gpu,
                    np.int32(len(query_tokens_ids_single)),
                    data_gpu,
                    indices_gpu,
                    indptr_gpu,
                    candidate_scores,
                ),
            )
        else:
            kernel(
                (grid_size,),
                (block_size,),
                (
                    candidates,
                    np.int32(n_candidates),
                    query_terms_gpu,
                    np.int32(len(query_tokens_ids_single)),
                    data_gpu,
                    indices_gpu,
                    indptr_gpu,
                    candidate_scores,
                ),
            )

        topk_scores, topk_docs_pos = _topk_cupy_gpu(
            candidate_scores, k=k, sorted=True
        )
        topk_docs = candidates[topk_docs_pos].astype(int_dtype, copy=False)
        outside_bound = (
            float(term_max[dropped_terms].sum()) if len(dropped_terms) else 0.0
        )

        retrieved_docs[query_pos] = topk_docs
        retrieved_scores[query_pos] = topk_scores.astype(dtype, copy=False)
        cert_candidate_positions.append(query_pos)
        cert_topk_scores.append(topk_scores)
        cert_outside_bounds.append(outside_bound)

    fallback_positions.extend(
        _resolve_candidate_certification(
            candidate_positions=cert_candidate_positions,
            topk_scores=cert_topk_scores,
            outside_bounds=cert_outside_bounds,
        )
    )
    for query_pos in fallback_positions:
        retrieved_docs[query_pos] = None
        retrieved_scores[query_pos] = None

    if fallback_positions:
        fallback_docs, fallback_scores = _retrieve_dense_batch_cupy(
            [query_tokens_ids[pos] for pos in fallback_positions],
            scores,
            k=k,
            sorted=True,
            dtype=dtype,
            int_dtype=int_dtype,
        )
        for fallback_idx, query_pos in enumerate(fallback_positions):
            retrieved_docs[query_pos] = cupy.asarray(
                fallback_docs[fallback_idx], dtype=int_dtype
            )
            retrieved_scores[query_pos] = cupy.asarray(
                fallback_scores[fallback_idx], dtype=dtype
            )

    return (
        cupy.asnumpy(cupy.stack(retrieved_docs)),
        cupy.asnumpy(cupy.stack(retrieved_scores)),
    )


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
    global _WARNED_IGNORED_CHUNKSIZE, _WARNED_IGNORED_N_THREADS
    cupy = _require_cupy()

    if backend_selection == "auto":
        backend_selection = "cupy"

    if backend_selection != "cupy":
        error_msg = (
            "The `cupy` backend must use backend_selection='cupy' or 'auto'. "
            "Please choose a different backend or change backend_selection to cupy."
        )
        raise ValueError(error_msg)

    if chunksize is not None and not _WARNED_IGNORED_CHUNKSIZE:
        logging.warning(
            "The `chunksize` parameter is ignored in the `retrieve` function "
            "when using the `cupy` backend."
        )
        _WARNED_IGNORED_CHUNKSIZE = True

    if n_threads not in (0, 1) and not _WARNED_IGNORED_N_THREADS:
        logging.warning(
            "The `n_threads` parameter is ignored in the `retrieve` function "
            "when using the `cupy` backend."
        )
        _WARNED_IGNORED_N_THREADS = True

    allowed_return_as = ["tuple", "documents"]
    if return_as not in allowed_return_as:
        raise ValueError("`return_as` must be either 'tuple' or 'documents'")

    dtype = np.dtype(dtype)
    int_dtype = np.dtype(int_dtype)
    num_queries = len(query_tokens_ids)

    cache = _get_cupy_index_cache(scores, dtype=dtype, int_dtype=int_dtype)
    data_gpu = cache["data"]
    indices_gpu = cache["indices"]
    indptr = cache["indptr_cpu"]

    if nonoccurrence_array is not None:
        nonoccurrence_array_gpu = cupy.asarray(nonoccurrence_array, dtype=dtype)
    else:
        nonoccurrence_array_gpu = None

    if weight_mask is not None:
        weight_mask_gpu = cupy.asarray(weight_mask)
    else:
        weight_mask_gpu = None

    sparse_result = None
    if sorted and nonoccurrence_array is None and weight_mask is None:
        max_postings = int(
            os.environ.get("BM25S_CUPY_SPARSE_MAX_POSTINGS", "50000000")
        )
        sparse_result = _retrieve_sparse_batch_cupy(
            query_tokens_ids=query_tokens_ids,
            scores=scores,
            k=k,
            dtype=dtype,
            int_dtype=int_dtype,
            max_postings=max_postings,
        )

    if sparse_result is not None:
        retrieved_indices, retrieved_scores = sparse_result
    else:
        candidate_result = None
        if sorted and nonoccurrence_array is None and weight_mask is None:
            df_threshold = int(
                os.environ.get("BM25S_CUPY_CANDIDATE_DF_THRESHOLD", "75000")
            )
            if df_threshold > 0:
                candidate_result = _retrieve_candidate_batch_cupy(
                    query_tokens_ids=query_tokens_ids,
                    scores=scores,
                    k=k,
                    dtype=dtype,
                    int_dtype=int_dtype,
                    df_threshold=df_threshold,
                )

        if candidate_result is not None:
            retrieved_indices, retrieved_scores = candidate_result
        else:
            retrieved_indices, retrieved_scores = _retrieve_dense_batch_cupy(
                query_tokens_ids=query_tokens_ids,
                scores=scores,
                k=k,
                sorted=sorted,
                dtype=dtype,
                int_dtype=int_dtype,
                nonoccurrence_array=nonoccurrence_array_gpu,
                weight_mask=weight_mask_gpu,
            )

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
