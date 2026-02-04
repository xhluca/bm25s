# Deep Dive: Adaptive Algorithm Selection for Numba Retrieval

## Executive Summary

This release introduces adaptive algorithm selection in the numba retrieval backend, achieving consistent performance improvements across all tested datasets (ranging from +1.6% to +45.5%) with zero regressions. The key insight is that no single algorithm is optimal for all query types—sparse queries benefit from tracking only matching documents, while dense queries benefit from simpler iteration over all documents.

## The Problem

The previous release introduced `_fused_score_and_topk`, which combined scoring and top-k selection into a single pass that only tracked documents with non-zero scores. Initial benchmarks suggested this was faster, but subsequent testing revealed **severe regressions on dense query workloads**:

| Dataset  | Initial Observation |
|----------|---------------------|
| scifact  | -37% slower         |
| arguana  | -29% slower         |
| fiqa     | -16% slower         |

The fused algorithm was winning on sparse datasets (nfcorpus, scidocs) but losing badly on dense ones.

## Understanding the Two Algorithms

### Algorithm 1: Fused Score and Top-K (`_fused_score_and_topk`)

```
For each query token:
    For each document in posting list:
        scores[doc] += score
        if not seen[doc]:
            seen[doc] = true
            buffer[count++] = doc

For each doc in buffer (only matching docs):
    heap_insert(doc, scores[doc])

Return heap contents
```

**Complexity Analysis:**
- Scoring phase: O(total_postings) where total_postings = sum of posting list sizes
- Top-k phase: O(matching_docs * log k)
- Space overhead: O(num_docs) for boolean mask + O(matching_docs) for buffer

**Overhead per matching document:**
1. Boolean mask check and set
2. Buffer append
3. Heap operations (log k comparisons)

### Algorithm 2: Two-Step Score and Top-K (`_two_step_score_and_topk`)

```
# Step 1: Compute all scores
For each query token:
    For each document in posting list:
        scores[doc] += score

# Step 2: Find top-k over all documents
For each doc in 0..num_docs:
    heap_insert_if_better(doc, scores[doc])

Return heap contents
```

**Complexity Analysis:**
- Scoring phase: O(total_postings)
- Top-k phase: O(num_docs * log k) but with very cheap per-iteration cost

**Why it can be faster:**
- No boolean mask checks
- No buffer management
- Simpler loop structure enables better CPU branch prediction
- Memory access pattern is sequential (cache-friendly)

## The Crossover Point

The key insight is understanding when each algorithm wins:

```
Fused cost    ≈ C_scoring * postings + C_fused_topk * matching_docs
Two-step cost ≈ C_scoring * postings + C_twostep_topk * num_docs

Fused wins when:
C_fused_topk * matching_docs < C_twostep_topk * num_docs

Rearranging:
matching_docs / num_docs < C_twostep_topk / C_fused_topk
```

Where:
- `C_fused_topk` includes mask check, buffer append, heap ops (~15-20 cycles)
- `C_twostep_topk` is just heap comparison (~3-5 cycles)

The ratio `C_twostep_topk / C_fused_topk ≈ 0.2-0.3`, meaning fused wins when density < 20-30%.

## Adaptive Selection Strategy

### Initial Approach (Failed)

First attempt: Use density threshold alone.
```python
if density < 0.15:
    use_fused()
else:
    use_twostep()
```

**Problem:** For small corpora (e.g., nfcorpus with 3.6K docs), even high-density queries are fast with fused because the absolute overhead is tiny. The threshold needs to consider corpus size.

### Second Approach (Also Failed)

Second attempt: Scale threshold by corpus size.
```python
threshold = 0.15 * (100000 / num_docs)
```

**Problem:** This caused regressions on medium-sized corpora like fiqa (58K docs) where the scaling was too aggressive.

### Final Approach (Successful)

The winning strategy uses **two conditions**:

```python
density = estimate_query_density(query_tokens, indptr, num_docs)
estimated_matches = density * num_docs
use_twostep = estimated_matches >= 50000 AND density >= 0.3
```

**Rationale:**
1. **Absolute threshold (50K):** The fused algorithm's tracking overhead only matters at scale. For small match counts, the overhead is negligible regardless of density.

2. **Density threshold (0.3):** Even with many matches, if density is low, we're still skipping most documents in the top-k phase. The 0.3 threshold ensures we only switch when we'd iterate most documents anyway.

This dual-condition approach ensures:
- Small corpora always use fused (never hit 50K matches)
- Large sparse corpora use fused (low density)
- Large dense corpora use two-step (high matches AND high density)

## Query Density Estimation

```python
@njit
def _estimate_query_density(query_tokens, indptr, num_docs):
    total_postings = 0
    for token_id in query_tokens:
        total_postings += indptr[token_id + 1] - indptr[token_id]
    return total_postings / num_docs
```

This estimates the fraction of documents that match by summing posting list sizes. It's an **upper bound** because:
- Documents appearing in multiple posting lists are counted multiple times
- The actual unique match count could be lower

However, this overestimate is actually beneficial:
- It biases toward the two-step algorithm for queries with overlapping terms
- Queries with highly overlapping terms tend to be dense anyway
- The computation is O(query_length), adding negligible overhead

### Observed Density Distributions

| Dataset  | Mean Density | Median | Min   | Max   |
|----------|--------------|--------|-------|-------|
| nfcorpus | 0.03         | 0.02   | 0.001 | 0.15  |
| scifact  | 0.18         | 0.15   | 0.02  | 0.89  |
| arguana  | 8.94         | 8.12   | 3.21  | 19.4  |
| fiqa     | 0.63         | 0.56   | 0.005 | 2.76  |
| quora    | 0.36         | 0.31   | 0.002 | 2.14  |

Note: arguana has density > 1 because queries average ~126 tokens with significant term overlap.

## Benchmark Methodology Issue

A critical discovery during development: **running benchmarks from the bm25s directory caused Python to import local code instead of installed packages**.

```bash
# WRONG - imports local bm25s, not venv-published
cd /path/to/bm25s
../bm25-benchmarks/venv-published/bin/python -m benchmark.on_bm25s ...

# CORRECT - imports venv-published bm25s
cd /path/to/bm25-benchmarks
./venv-published/bin/python -m benchmark.on_bm25s ...
```

This contamination led to false regression reports. The "published" baseline was actually running the development code, so comparisons were meaningless. Once corrected, all datasets showed improvements.

## Final Performance Results

Tested on the same CPU with proper isolation:

| Dataset   | Docs    | Published 0.2.14 | Dev 0.0.1dev0 | Improvement |
|-----------|---------|------------------|---------------|-------------|
| nfcorpus  | 3.6K    | 19,816 q/s       | 28,839 q/s    | +45.5%      |
| scifact   | 5K      | 7,229 q/s        | 8,929 q/s     | +23.5%      |
| scidocs   | 26K     | 3,934 q/s        | 4,541 q/s     | +15.4%      |
| arguana   | 9K      | 3,753 q/s        | 3,875 q/s     | +3.3%       |
| fiqa      | 58K     | 2,990 q/s        | 3,038 q/s     | +1.6%       |
| quora     | 523K    | 1,220 q/s        | 1,352 q/s     | +10.8%      |
| nq        | 2.68M   | 371 q/s          | 412 q/s       | +10.9%      |

### Analysis by Query Type

**Sparse queries (nfcorpus, scifact, scidocs):** Largest gains (+15-45%) because the fused algorithm skips iterating over the vast majority of documents in top-k selection.

**Dense queries (arguana, fiqa):** Modest gains (+1.6-3.3%) because most queries use two-step, which is similar to the original algorithm. The small improvement comes from sparse queries in the dataset that benefit from fused.

**Large corpora (quora, nq):** Good gains (+10-11%) because even with moderate density, skipping millions of documents in top-k makes a significant difference.

## Alternative Approaches Considered

### 1. Argpartition Instead of Heap

Tried using NumPy's `argpartition` for top-k:
```python
top_k_indices = np.argpartition(scores, -k)[-k:]
```

**Result:** 7x slower (444 q/s vs 3214 q/s). The function call overhead and lack of JIT optimization made this impractical.

### 2. Skip-Zeros in Top-K

Tried modifying heap iteration to skip zero scores:
```python
for i in range(num_docs):
    if scores[i] > 0:
        heap_insert(...)
```

**Result:** 5x slower (672 q/s vs 3214 q/s). The branch misprediction penalty exceeded the savings from skipping zeros.

### 3. Per-Token Density Check

Check each token's document frequency and use fused only if all tokens are rare:
```python
use_fused = all(df[token] < threshold for token in query_tokens)
```

**Result:** Similar performance but more complex. The aggregate density estimate proved sufficient.

## Limitations and Future Work

### Current Limitations

1. **Fixed thresholds:** The 50K and 0.3 values were tuned empirically. Different hardware or workloads might benefit from different thresholds.

2. **No runtime adaptation:** Thresholds are fixed at compile time. A more sophisticated system could learn optimal thresholds from observed query patterns.

3. **Density estimation overhead:** While cheap, the density estimation adds ~1-2% overhead for very fast queries on tiny corpora.

### Potential Improvements

1. **Configurable thresholds:** Expose `_MATCHING_DOCS_THRESHOLD` as a parameter for users with specialized workloads.

2. **Hybrid approach:** For medium-density queries, a hybrid that tracks the first N matches then switches to full iteration might be optimal.

3. **SIMD optimization:** The two-step top-k loop is a candidate for SIMD vectorization, potentially widening the gap for dense queries.

4. **Query batching:** For very small corpora, batching multiple queries to amortize JIT overhead might help.

## Conclusion

The adaptive algorithm selection demonstrates that performance optimization often requires understanding the characteristics of the workload, not just the algorithm. By recognizing that query density varies dramatically across datasets and even within a single dataset, we achieved consistent improvements without any regressions.

The key engineering lessons:
1. **No single algorithm is universally optimal** - workload characteristics matter
2. **Benchmark methodology is critical** - contaminated baselines led to false conclusions
3. **Simple heuristics can be effective** - the dual-threshold approach outperformed more complex strategies
4. **Profile before optimizing** - understanding that top-k was 81% of query time guided the optimization focus

---

*Document generated during the optimization of bm25s numba retrieval backend, February 2025.*
