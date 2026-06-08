#!/usr/bin/env python3
"""Synthetic throughput benchmark for the CuPy sparse BM25 fast path.

This benchmark isolates BM25 retrieval only. It does not tokenize text and does
not build an index; it generates a CSC score matrix directly so backend timing is
not mixed with data preparation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bm25s.cupy.retrieve_utils import _retrieve_cupy_functional
from bm25s.numba.retrieve_utils import _retrieve_numba_functional


def now_utc_slug() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def run_cmd(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def make_scores(num_docs: int, vocab_size: int, postings_per_term: int, seed: int):
    rng = np.random.default_rng(seed)
    indices_parts = []
    data_parts = []
    indptr = np.empty(vocab_size + 1, dtype=np.int32)
    cursor = 0
    for term_id in range(vocab_size):
        indptr[term_id] = cursor
        docs = rng.choice(num_docs, size=postings_per_term, replace=False).astype(
            np.int32
        )
        docs.sort()
        values = rng.random(postings_per_term, dtype=np.float32) + np.float32(0.01)
        indices_parts.append(docs)
        data_parts.append(values)
        cursor += postings_per_term
    indptr[vocab_size] = cursor
    return {
        "data": np.concatenate(data_parts).astype(np.float32),
        "indices": np.concatenate(indices_parts).astype(np.int32),
        "indptr": indptr,
        "num_docs": num_docs,
    }


def make_queries(num_queries: int, vocab_size: int, query_len: int, seed: int):
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, vocab_size, size=query_len, dtype=np.int32)
        for _ in range(num_queries)
    ]


def sync_gpu() -> None:
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def gpu_snapshot() -> dict | None:
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "gpu_mem_free_mib": int(free // 1024 // 1024),
            "gpu_mem_total_mib": int(total // 1024 // 1024),
            "cupy_pool_used_mib": int(
                cp.get_default_memory_pool().used_bytes() // 1024 // 1024
            ),
        }
    except Exception:
        return None


def timed(name: str, fn, repeat: int, sync: bool):
    samples = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        if sync:
            sync_gpu()
        samples.append(time.perf_counter() - t0)
    best = min(samples)
    return {
        "name": name,
        "elapsed_s_best": best,
        "elapsed_s_samples": samples,
        "result": result,
    }


def compare_docs(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        return {
            "shape_mismatch": True,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    overlaps = []
    for left_row, right_row in zip(left, right):
        left_set = set(int(x) for x in left_row)
        right_set = set(int(x) for x in right_row)
        overlaps.append(len(left_set & right_set) / max(len(left_set), 1))
    diff = left != right
    return {
        "shape_mismatch": False,
        "queries_with_different_order": int(diff.any(axis=1).sum()),
        "different_positions": int(diff.sum()),
        "mean_overlap_fraction": float(np.mean(overlaps)),
        "min_overlap_fraction": float(np.min(overlaps)),
    }


def write_report(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )

    rows = payload["results"]
    lines = [
        "# CuPy Sparse Fast Path Benchmark",
        "",
        f"created_at_utc: `{payload['created_at_utc']}`",
        f"bm25s_commit: `{payload['repo']['commit']}`",
        f"num_docs: `{payload['workload']['num_docs']}`",
        f"vocab_size: `{payload['workload']['vocab_size']}`",
        f"postings_per_term: `{payload['workload']['postings_per_term']}`",
        f"num_queries: `{payload['workload']['num_queries']}`",
        f"query_len: `{payload['workload']['query_len']}`",
        f"topk: `{payload['workload']['topk']}`",
        "",
        "| backend | best_s | qps | speedup_vs_numba | speedup_vs_cupy_dense |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {elapsed:.6f} | {qps:.2f} | {numba:.3f}x | {dense:.3f}x |".format(
                name=row["name"],
                elapsed=row["elapsed_s_best"],
                qps=row["queries_per_second"],
                numba=row.get("speedup_vs_numba") or 0.0,
                dense=row.get("speedup_vs_cupy_dense") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "```json",
            json.dumps(payload["correctness"], indent=2),
            "```",
            "",
        ]
    )
    (output_dir / "benchmark_summary.md").write_text("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results") / f"cupy_sparse_fastpath_{now_utc_slug()}")
    p.add_argument("--num-docs", type=int, default=200_000)
    p.add_argument("--vocab-size", type=int, default=20_000)
    p.add_argument("--postings-per-term", type=int, default=160)
    p.add_argument("--num-queries", type=int, default=256)
    p.add_argument("--query-len", type=int, default=16)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    scores = make_scores(
        args.num_docs, args.vocab_size, args.postings_per_term, args.seed
    )
    queries = make_queries(
        args.num_queries, args.vocab_size, args.query_len, args.seed + 1
    )

    def run_numba():
        return _retrieve_numba_functional(
            queries,
            scores,
            corpus=None,
            k=args.topk,
            sorted=True,
            return_as="tuple",
            show_progress=False,
            n_threads=32,
            backend_selection="numba",
            dtype="float32",
            int_dtype="int32",
        )

    def run_cupy_dense():
        old = os.environ.get("BM25S_CUPY_SPARSE_MAX_POSTINGS")
        os.environ["BM25S_CUPY_SPARSE_MAX_POSTINGS"] = "0"
        try:
            return _retrieve_cupy_functional(
                queries,
                scores,
                corpus=None,
                k=args.topk,
                sorted=True,
                return_as="tuple",
                show_progress=False,
                backend_selection="cupy",
                dtype="float32",
                int_dtype="int32",
            )
        finally:
            if old is None:
                os.environ.pop("BM25S_CUPY_SPARSE_MAX_POSTINGS", None)
            else:
                os.environ["BM25S_CUPY_SPARSE_MAX_POSTINGS"] = old

    def run_cupy_sparse():
        old = os.environ.get("BM25S_CUPY_SPARSE_MAX_POSTINGS")
        os.environ["BM25S_CUPY_SPARSE_MAX_POSTINGS"] = str(10**12)
        try:
            return _retrieve_cupy_functional(
                queries,
                scores,
                corpus=None,
                k=args.topk,
                sorted=True,
                return_as="tuple",
                show_progress=False,
                backend_selection="cupy",
                dtype="float32",
                int_dtype="int32",
            )
        finally:
            if old is None:
                os.environ.pop("BM25S_CUPY_SPARSE_MAX_POSTINGS", None)
            else:
                os.environ["BM25S_CUPY_SPARSE_MAX_POSTINGS"] = old

    # Warm compilation and GPU memory pools outside the timed samples.
    numba_docs, numba_scores = run_numba()
    dense_docs, dense_scores = run_cupy_dense()
    sparse_docs, sparse_scores = run_cupy_sparse()
    sync_gpu()

    measurements = [
        timed("numba", run_numba, args.repeat, sync=False),
        timed("cupy_dense_fallback", run_cupy_dense, args.repeat, sync=True),
        timed("cupy_sparse_fastpath", run_cupy_sparse, args.repeat, sync=True),
    ]

    base_numba = measurements[0]["elapsed_s_best"]
    base_dense = measurements[1]["elapsed_s_best"]
    results = []
    for item in measurements:
        elapsed = item["elapsed_s_best"]
        row = {
            "name": item["name"],
            "elapsed_s_best": elapsed,
            "elapsed_s_samples": item["elapsed_s_samples"],
            "queries_per_second": args.num_queries / elapsed,
            "speedup_vs_numba": base_numba / elapsed,
            "speedup_vs_cupy_dense": base_dense / elapsed,
        }
        results.append(row)

    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "repo": {
            "cwd": str(Path.cwd()),
            "commit": run_cmd(["git", "rev-parse", "HEAD"], cwd=Path.cwd()),
            "branch": run_cmd(["git", "branch", "--show-current"], cwd=Path.cwd()),
            "status": run_cmd(["git", "status", "--short"], cwd=Path.cwd()),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.executable,
            "cpu_count": os.cpu_count(),
            "gpu": run_cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "gpu_after": gpu_snapshot(),
        },
        "workload": {
            "num_docs": args.num_docs,
            "vocab_size": args.vocab_size,
            "postings_per_term": args.postings_per_term,
            "num_queries": args.num_queries,
            "query_len": args.query_len,
            "topk": args.topk,
            "repeat": args.repeat,
            "seed": args.seed,
        },
        "results": results,
        "correctness": {
            "numba_vs_cupy_dense_docs": compare_docs(numba_docs, dense_docs),
            "numba_vs_cupy_sparse_docs": compare_docs(numba_docs, sparse_docs),
            "cupy_dense_vs_sparse_docs": compare_docs(dense_docs, sparse_docs),
            "numba_vs_cupy_dense_scores_max_abs_diff": float(
                np.max(np.abs(numba_scores - dense_scores))
            ),
            "numba_vs_cupy_sparse_scores_max_abs_diff": float(
                np.max(np.abs(numba_scores - sparse_scores))
            ),
            "cupy_dense_vs_sparse_scores_max_abs_diff": float(
                np.max(np.abs(dense_scores - sparse_scores))
            ),
        },
    }
    write_report(args.output_dir, payload)
    print(f"[bench] wrote {args.output_dir / 'benchmark_results.json'}", flush=True)
    print(f"[bench] wrote {args.output_dir / 'benchmark_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
