# BM25S Performance Analysis

## Summary

Benchmarking the dev version (0.0.1dev0) against published (0.2.14) shows that the adaptive algorithm in `bm25s/numba/retrieve_utils.py` provides consistent improvements across all tested datasets.

## Benchmark Results (Proper Baseline Comparison)

| Dataset   | Docs    | Query Type           | Published 0.2.14 (q/s) | Dev 0.0.1dev0 (q/s) | Improvement |
|-----------|---------|----------------------|------------------------|---------------------|-------------|
| nfcorpus  | 3.6K    | Medical terms        | 19,816                 | 28,839              | **+45.5%**  |
| scifact   | 5K      | Scientific claims    | 7,229                  | 8,929               | **+23.5%**  |
| scidocs   | 26K     | Scientific terms     | 3,934                  | 4,541               | **+15.4%**  |
| arguana   | 9K      | Debate arguments     | 3,753                  | 3,875               | **+3.3%**   |
| fiqa      | 58K     | Financial Q&A        | 2,990                  | 3,038               | **+1.6%**   |
| quora     | 523K    | General questions    | 1,220                  | 1,352               | **+10.8%**  |
| nq        | 2.68M   | Entity queries       | 371                    | 412                 | **+10.9%**  |

## Adaptive Algorithm Selection

The implementation uses two algorithms and selects based on query density:

### Fused Algorithm (`_fused_score_and_topk`)
- Tracks documents with non-zero scores using a mask + buffer
- Uses a min-heap for top-k selection
- Only iterates over matching documents for top-k
- **Best for sparse queries** where `matching_docs << total_docs`

### Two-Step Algorithm (`_two_step_score_and_topk`)
- Computes all scores: O(num_docs)
- Heap-based top-k selection: O(num_docs)
- Lower constant overhead per document
- **Best for dense queries** where `matching_docs ≈ total_docs`

### Selection Criteria
```python
density = _estimate_query_density(query_tokens, indptr, num_docs)
estimated_matches = density * num_docs
use_twostep = estimated_matches >= 50000 and density >= 0.3
```

The algorithm switches to two-step only when BOTH:
1. Estimated matches >= 50,000 (tracking overhead matters at scale)
2. Density >= 30% (iterating all docs isn't much worse)

## Why Sparse Queries Benefit More

Medical/scientific terms like "cholesterol", "asthma" match few documents:
- Fused algorithm skips O(num_docs - matching_docs) in top-k phase
- The savings outweigh the mask/buffer tracking overhead

For general queries with common words ("how", "what", "why"):
- Most documents match, so the fused algorithm's tracking overhead doesn't pay off
- The adaptive selection correctly uses the two-step algorithm

## Key Files

- `bm25s/numba/retrieve_utils.py`: Contains both algorithms and adaptive selection logic
- `bm25s/scoring.py`: Contains `_compute_relevance_from_scores_jit_ready`
- `bm25s/numba/selection.py`: Contains `_numba_sorted_top_k`

## Test Commands

```bash
# Run benchmark from bm25-benchmarks directory (IMPORTANT: not from bm25s directory!)
cd /path/to/bm25-benchmarks

# Test with published version
./venv-published/bin/python -m benchmark.on_bm25s -d <dataset> --scorers jit --backends numba

# Test with dev version
./venv-devel/bin/python -m benchmark.on_bm25s -d <dataset> --scorers jit --backends numba

# Datasets to test (mix of sparse and dense):
# Sparse: nfcorpus, scidocs, scifact
# Dense: quora, arguana, fiqa
# Large: nq, msmarco
```

## Important Note on Baseline Comparison

When running benchmarks, always run from the `bm25-benchmarks` directory, NOT from the `bm25s` directory. Running from the bm25s directory causes Python to import the local development code instead of the installed version, leading to incorrect baseline comparisons.
