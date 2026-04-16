from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

try:  # pragma: no cover - environment-specific
    import faiss  # type: ignore
except Exception:  # pragma: no cover - environment-specific
    faiss = None


@dataclass(slots=True)
class SearchHit:
    rank: int
    index: int
    distance: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class BuiltSearchIndex:
    method: str
    metric: str
    config: dict[str, Any]
    vector_count: int
    dim: int
    build_time_ms: float
    memory_bytes_est: int | None
    backend: str
    base_vectors: np.ndarray | None = None
    faiss_index: Any = None
    _cosine_normalized: bool = False


def faiss_available() -> bool:
    return faiss is not None


def available_methods() -> list[str]:
    return ["brute", "faiss_flat", "faiss_hnsw", "faiss_ivf", "faiss_ivfpq"]


def _as_f32_contig(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = _as_f32_contig(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = _ensure_2d(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _topk_largest(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_queries, n_items = scores.shape
    k = max(0, min(int(k), n_items))
    if k == 0:
        return (
            np.zeros((n_queries, 0), dtype=np.float32),
            np.zeros((n_queries, 0), dtype=np.int64),
        )
    if k == n_items:
        idx = np.argsort(-scores, axis=1)
    else:
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        part_scores = scores[rows, part]
        order = np.argsort(-part_scores, axis=1)
        idx = part[rows, order]
    rows = np.arange(n_queries)[:, None]
    top = scores[rows, idx].astype(np.float32, copy=False)
    return top, idx.astype(np.int64, copy=False)


def _topk_smallest(dists: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_queries, n_items = dists.shape
    k = max(0, min(int(k), n_items))
    if k == 0:
        return (
            np.zeros((n_queries, 0), dtype=np.float32),
            np.zeros((n_queries, 0), dtype=np.int64),
        )
    if k == n_items:
        idx = np.argsort(dists, axis=1)
    else:
        part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        part_dists = dists[rows, part]
        order = np.argsort(part_dists, axis=1)
        idx = part[rows, order]
    rows = np.arange(n_queries)[:, None]
    top = dists[rows, idx].astype(np.float32, copy=False)
    return top, idx.astype(np.int64, copy=False)


def _prepare_vectors(vectors: np.ndarray, metric: str) -> tuple[np.ndarray, bool]:
    x = _ensure_2d(vectors)
    if metric == "cosine":
        return _normalize_rows(x), True
    if metric == "l2":
        return x, False
    raise ValueError(f"Unsupported metric: {metric}")


def _estimate_faiss_memory(index: Any) -> int | None:
    if faiss is None or index is None:
        return None
    try:  # pragma: no cover - faiss optional
        return int(faiss.serialize_index(index).nbytes)
    except Exception:
        return None


def build_index(
    method: str,
    vectors: np.ndarray,
    *,
    metric: str = "cosine",
    config: dict[str, Any] | None = None,
) -> BuiltSearchIndex:
    """Build a search index. Returns a reusable index object.

    `method` supports:
    - `brute` (exact baseline)
    - `faiss_flat` (exact in FAISS)
    - `faiss_hnsw` (scaffold + optional implementation if FAISS is available)
    - `faiss_ivf` (scaffold + optional implementation if FAISS is available)
    """

    cfg = dict(config or {})
    method = method.strip().lower()
    metric = metric.strip().lower()
    x, normalized = _prepare_vectors(vectors, metric)
    n, d = x.shape

    t0 = perf_counter()
    if method in {"brute", "brute_force"}:
        build_ms = (perf_counter() - t0) * 1000.0
        return BuiltSearchIndex(
            method="brute",
            metric=metric,
            config=cfg,
            vector_count=n,
            dim=d,
            build_time_ms=build_ms,
            memory_bytes_est=int(x.nbytes),
            backend="numpy",
            base_vectors=x,
            _cosine_normalized=normalized,
        )

    if faiss is None:
        raise RuntimeError(f"{method} requested but FAISS is not available.")

    if method == "faiss_flat":
        if metric == "l2":
            idx = faiss.IndexFlatL2(d)
        else:
            idx = faiss.IndexFlatIP(d)
        idx.add(x)
        build_ms = (perf_counter() - t0) * 1000.0
        return BuiltSearchIndex(
            method=method,
            metric=metric,
            config=cfg,
            vector_count=n,
            dim=d,
            build_time_ms=build_ms,
            memory_bytes_est=_estimate_faiss_memory(idx) or int(x.nbytes),
            backend="faiss",
            faiss_index=idx,
            _cosine_normalized=normalized,
        )

    if method == "faiss_hnsw":
        m = int(cfg.get("M", 32))
        ef_construction = int(cfg.get("ef_construction", 200))
        ef_search = int(cfg.get("ef_search", 64))
        metric_id = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        try:  # pragma: no cover - faiss-version dependent
            idx = faiss.IndexHNSWFlat(d, m, metric_id)
        except TypeError:  # pragma: no cover
            if metric != "l2":
                raise RuntimeError("Current FAISS build does not support HNSW cosine mode.")
            idx = faiss.IndexHNSWFlat(d, m)
        idx.hnsw.efConstruction = ef_construction
        idx.hnsw.efSearch = ef_search
        idx.add(x)
        build_ms = (perf_counter() - t0) * 1000.0
        return BuiltSearchIndex(
            method=method,
            metric=metric,
            config={"M": m, "ef_construction": ef_construction, "ef_search": ef_search},
            vector_count=n,
            dim=d,
            build_time_ms=build_ms,
            memory_bytes_est=_estimate_faiss_memory(idx) or int(x.nbytes),
            backend="faiss",
            faiss_index=idx,
            _cosine_normalized=normalized,
        )

    if method == "faiss_ivf":
        nlist = int(cfg.get("nlist", max(1, min(256, int(np.sqrt(max(n, 1)))))))
        nlist = max(1, min(nlist, max(1, n)))
        nprobe = int(cfg.get("nprobe", min(8, nlist)))
        if metric == "l2":
            quantizer = faiss.IndexFlatL2(d)
            metric_id = faiss.METRIC_L2
        else:
            quantizer = faiss.IndexFlatIP(d)
            metric_id = faiss.METRIC_INNER_PRODUCT
        idx = faiss.IndexIVFFlat(quantizer, d, nlist, metric_id)
        if not idx.is_trained:
            idx.train(x)
        idx.add(x)
        idx.nprobe = max(1, min(nprobe, nlist))
        build_ms = (perf_counter() - t0) * 1000.0
        return BuiltSearchIndex(
            method=method,
            metric=metric,
            config={"nlist": nlist, "nprobe": idx.nprobe},
            vector_count=n,
            dim=d,
            build_time_ms=build_ms,
            memory_bytes_est=_estimate_faiss_memory(idx) or int(x.nbytes),
            backend="faiss",
            faiss_index=idx,
            _cosine_normalized=normalized,
        )

    if method == "faiss_ivfpq":
        nlist = int(cfg.get("nlist", max(1, min(256, int(np.sqrt(max(n, 1)))))))
        nlist = max(1, min(nlist, max(1, n)))
        nprobe = int(cfg.get("nprobe", min(8, nlist)))
        pq_m = int(cfg.get("pq_m", 8))
        pq_nbits = int(cfg.get("pq_nbits", 8))

        # Keep pq_m valid for current embedding dimension.
        if d <= 0:
            raise ValueError("Invalid vector dimension for IVFPQ.")
        if pq_m <= 0:
            pq_m = 1
        if d % pq_m != 0:
            divisors = [v for v in range(1, d + 1) if d % v == 0]
            pq_m = min(divisors, key=lambda v: abs(v - pq_m))
        max_nbits = int(np.floor(np.log2(max(2, n))))
        pq_nbits = max(1, min(12, pq_nbits, max_nbits))

        if metric == "l2":
            quantizer = faiss.IndexFlatL2(d)
            metric_id = faiss.METRIC_L2
        else:
            quantizer = faiss.IndexFlatIP(d)
            metric_id = faiss.METRIC_INNER_PRODUCT

        try:  # pragma: no cover - faiss-version dependent
            idx = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits, metric_id)
        except TypeError:  # pragma: no cover
            if metric != "l2":
                raise RuntimeError("Current FAISS build does not support IVFPQ cosine mode.")
            idx = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits)
        if not idx.is_trained:
            idx.train(x)
        idx.add(x)
        idx.nprobe = max(1, min(nprobe, nlist))
        build_ms = (perf_counter() - t0) * 1000.0
        return BuiltSearchIndex(
            method=method,
            metric=metric,
            config={"nlist": nlist, "nprobe": idx.nprobe, "pq_m": pq_m, "pq_nbits": pq_nbits},
            vector_count=n,
            dim=d,
            build_time_ms=build_ms,
            memory_bytes_est=_estimate_faiss_memory(idx) or int(x.nbytes),
            backend="faiss",
            faiss_index=idx,
            _cosine_normalized=normalized,
        )

    raise ValueError(f"Unknown search method: {method}")


def query_index(index: BuiltSearchIndex, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    q = _ensure_2d(query_vec)
    if q.shape[1] != index.dim:
        raise ValueError(f"Query dim {q.shape[1]} != index dim {index.dim}")

    if index.metric == "cosine":
        q = _normalize_rows(q)

    k = max(0, min(int(k), index.vector_count))
    if k == 0:
        n = q.shape[0]
        return (
            np.zeros((n, 0), dtype=np.float32),
            np.zeros((n, 0), dtype=np.int64),
        )

    if index.backend == "numpy":
        base = index.base_vectors
        if base is None:
            raise RuntimeError("Brute-force index missing base vectors.")
        if index.metric == "cosine":
            scores = q @ base.T
            top_scores, idx = _topk_largest(scores, k)
            dists = (1.0 - top_scores).astype(np.float32, copy=False)
            return dists, idx
        q_norm = (q * q).sum(axis=1, keepdims=True)
        b_norm = (base * base).sum(axis=1, keepdims=True).T
        dists = np.maximum(q_norm + b_norm - 2.0 * (q @ base.T), 0.0)
        return _topk_smallest(dists, k)

    if index.backend == "faiss":
        dists, idx = index.faiss_index.search(q, k)
        dists = dists.astype(np.float32, copy=False)
        idx = idx.astype(np.int64, copy=False)
        if index.metric == "cosine":
            dists = 1.0 - dists
        return dists, idx

    raise RuntimeError(f"Unsupported backend: {index.backend}")


def hits_with_metadata(
    index: BuiltSearchIndex,
    query_vec: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    k: int,
    exclude_indices: set[int] | None = None,
    max_hits_per_game: int | None = None,
    min_ply_gap_same_game: int = 0,
    exclude_game_ids: set[str] | None = None,
    min_result_ply: int = 0,
    phase_filter: str | None = None,
    diversify_positions: bool = False,
) -> list[SearchHit]:
    candidate_k = min(index.vector_count, max(int(k) * (24 if diversify_positions else 12), int(k) + 120))
    d, i = query_index(index, query_vec, candidate_k)
    hits: list[SearchHit] = []
    excluded = exclude_indices or set()
    excluded_game_ids = exclude_game_ids or set()
    normalized_phase = (phase_filter or "").strip().lower()
    per_game_count: dict[str, int] = {}
    per_game_plies: dict[str, list[int]] = {}
    seen_position_signatures: set[str] = set()
    for dist, idx in zip(d[0], i[0], strict=False):
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= len(metadata) or idx_i in excluded:
            continue
        meta = metadata[idx_i]
        game_id = str(meta.get("game_id", ""))
        ply_index = int(meta.get("ply_index", 0))
        if game_id and game_id in excluded_game_ids:
            continue
        if ply_index < max(0, int(min_result_ply)):
            continue
        if normalized_phase and normalized_phase != "all":
            if str(meta.get("phase", "")).strip().lower() != normalized_phase:
                continue
        if game_id and max_hits_per_game is not None:
            if per_game_count.get(game_id, 0) >= max(1, int(max_hits_per_game)):
                continue
        if game_id and min_ply_gap_same_game > 0:
            used_plies = per_game_plies.get(game_id, [])
            if any(abs(ply_index - p) < int(min_ply_gap_same_game) for p in used_plies):
                continue
        if diversify_positions:
            fen = str(meta.get("fen", ""))
            placement_sig = fen.split(" ", 1)[0] if fen else f"idx:{idx_i}"
            if placement_sig in seen_position_signatures:
                continue

        hits.append(SearchHit(rank=len(hits) + 1, index=idx_i, distance=float(dist), metadata=meta))
        if game_id:
            per_game_count[game_id] = per_game_count.get(game_id, 0) + 1
            per_game_plies.setdefault(game_id, []).append(ply_index)
        if diversify_positions:
            seen_position_signatures.add(placement_sig)
        if len(hits) >= k:
            break
    return hits
