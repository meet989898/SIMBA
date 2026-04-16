from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from search import build_index, query_index


def _select_query_ids(n: int, query_count: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    q = min(n, max(1, int(query_count)))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=q, replace=False).astype(np.int64))


def _trim_self(indices: np.ndarray, query_ids: np.ndarray, k: int) -> np.ndarray:
    out = np.full((len(query_ids), k), -1, dtype=np.int64)
    for r, qid in enumerate(query_ids):
        w = 0
        for c in range(indices.shape[1]):
            idx = int(indices[r, c])
            if idx < 0 or idx == int(qid):
                continue
            out[r, w] = idx
            w += 1
            if w >= k:
                break
    return out


def _recall_at_k(gt: np.ndarray, approx: np.ndarray, k: int) -> float:
    if gt.shape[0] == 0 or k <= 0:
        return 0.0
    vals: list[float] = []
    for r in range(gt.shape[0]):
        g = {int(x) for x in gt[r, :k] if int(x) >= 0}
        a = {int(x) for x in approx[r, :k] if int(x) >= 0}
        if not g:
            continue
        vals.append(len(g & a) / float(len(g)))
    return float(np.mean(vals)) if vals else 0.0


def run_search_benchmark(
    vectors: np.ndarray,
    *,
    methods: list[str],
    metric: str = "cosine",
    k: int = 10,
    query_count: int = 50,
    seed: int = 42,
    method_configs: dict[str, dict[str, Any]] | None = None,
    compute_recall: bool = True,
) -> dict[str, Any]:
    """Run a lightweight benchmark over selected methods.

    Reports build time, query latency distribution (p50/p95/p99), and memory estimate.
    Recall@k is computed vs brute-force when requested.
    """

    x = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Benchmark requires a non-empty 2D vector matrix.")

    n, dim = x.shape
    qids = _select_query_ids(n, query_count, seed)
    qvecs = x[qids]
    k_eval = min(max(1, int(k)), max(1, n - 1))
    configs = method_configs or {}

    gt_neighbors: np.ndarray | None = None
    if compute_recall:
        brute = build_index("brute", x, metric=metric, config=configs.get("brute", {}))
        _, brute_idx = query_index(brute, qvecs, min(n, k_eval + 1))
        gt_neighbors = _trim_self(brute_idx, qids, k_eval)

    rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any] = {
            "method": method,
            "metric": metric,
            "n_vectors": int(n),
            "dim": int(dim),
            "query_count": int(len(qids)),
            "k": int(k_eval),
        }
        try:
            built = build_index(method, x, metric=metric, config=configs.get(method, {}))
            row["build_time_ms"] = built.build_time_ms
            row["memory_bytes_est"] = built.memory_bytes_est

            k_eff = min(n, k_eval + 1)
            lat_ms: list[float] = []
            seq_idx_rows: list[np.ndarray] = []
            for q in qvecs:
                t0 = perf_counter()
                _, idx = query_index(built, q, k_eff)
                lat_ms.append((perf_counter() - t0) * 1000.0)
                seq_idx_rows.append(idx[0])

            lat_arr = np.asarray(lat_ms, dtype=np.float64)
            row["query_latency_ms_mean"] = float(lat_arr.mean()) if lat_arr.size else 0.0
            row["query_latency_ms_std"] = float(lat_arr.std()) if lat_arr.size else 0.0
            row["query_latency_ms_p50"] = float(np.percentile(lat_arr, 50)) if lat_arr.size else 0.0
            row["query_latency_ms_p95"] = float(np.percentile(lat_arr, 95)) if lat_arr.size else 0.0
            row["query_latency_ms_p99"] = float(np.percentile(lat_arr, 99)) if lat_arr.size else 0.0
            row["query_qps_est"] = float(1000.0 / lat_arr.mean()) if lat_arr.size and float(lat_arr.mean()) > 0 else None

            if compute_recall and gt_neighbors is not None:
                approx_idx = np.vstack(seq_idx_rows) if seq_idx_rows else np.zeros((0, 0), dtype=np.int64)
                approx_trim = _trim_self(approx_idx, qids, k_eval)
                row["recall_at_k"] = _recall_at_k(gt_neighbors, approx_trim, k_eval)
                row["recall_at_1"] = _recall_at_k(gt_neighbors, approx_trim, 1)
            else:
                row["recall_at_k"] = None
                row["recall_at_1"] = None
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    return {
        "dataset": {"n_vectors": int(n), "dim": int(dim)},
        "config": {
            "metric": metric,
            "k": int(k_eval),
            "query_count": int(len(qids)),
            "seed": int(seed),
            "compute_recall": bool(compute_recall),
        },
        "results": rows,
    }
