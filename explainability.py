from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np

from eval_engine import normalize_eval_for_embedding


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def score_from_distance(distance: float) -> float:
    return float(max(0.0, min(100.0, 100.0 * (1.0 - float(distance)))))


def default_score_bands() -> dict[str, tuple[float, float]]:
    return {
        "perfect": (95.0, 100.0),
        "high": (80.0, 94.99),
        "medium": (60.0, 79.99),
        "low": (0.0, 59.99),
    }


def normalize_score_bands(raw: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    out = default_score_bands()
    if not isinstance(raw, dict):
        return out
    for key in ("perfect", "high", "medium", "low"):
        val = raw.get(key)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            lo = float(val[0])
            hi = float(val[1])
            out[key] = (min(lo, hi), max(lo, hi))
    return out


def score_band(score: float, bands: dict[str, tuple[float, float]]) -> str:
    s = float(score)
    for name in ("perfect", "high", "medium", "low"):
        lo, hi = bands[name]
        if lo <= s <= hi:
            return name
    return "low"


def cosine_similarity_distance(query_vec: np.ndarray, candidate_vec: np.ndarray) -> tuple[float, float]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vec, dtype=np.float32).reshape(-1)
    if q.shape != c.shape:
        raise ValueError(f"Vector shape mismatch: {q.shape} vs {c.shape}")
    qn = np.linalg.norm(q)
    cn = np.linalg.norm(c)
    if qn == 0.0 or cn == 0.0:
        return 0.0, 1.0
    sim = float(np.dot(q, c) / (qn * cn))
    sim = float(max(-1.0, min(1.0, sim)))
    dist = float(max(0.0, min(2.0, 1.0 - sim)))
    return sim, dist


def group_contributions_cosine(
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    group_slices: dict[str, slice],
) -> list[dict[str, Any]]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vec, dtype=np.float32).reshape(-1)
    if q.shape != c.shape:
        raise ValueError(f"Vector shape mismatch: {q.shape} vs {c.shape}")

    qn = np.linalg.norm(q)
    cn = np.linalg.norm(c)
    if qn == 0.0 or cn == 0.0:
        return [{"group": name, "contribution": 0.0, "share_abs_pct": 0.0, "share_signed_pct": 0.0} for name in group_slices]

    q_norm = q / qn
    c_norm = c / cn
    total_similarity = float(np.dot(q_norm, c_norm))

    rows: list[dict[str, Any]] = []
    for group, slc in group_slices.items():
        contrib = float(np.dot(q_norm[slc], c_norm[slc]))
        rows.append({"group": group, "contribution": contrib})

    abs_sum = float(sum(abs(r["contribution"]) for r in rows))
    for row in rows:
        contrib = float(row["contribution"])
        row["share_abs_pct"] = 0.0 if abs_sum == 0.0 else (abs(contrib) / abs_sum) * 100.0
        row["share_signed_pct"] = 0.0 if total_similarity == 0.0 else (contrib / total_similarity) * 100.0
    return rows


def _material_snapshot(board: chess.Board) -> tuple[float, float, float]:
    white_total = 0.0
    black_total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        white_total += len(board.pieces(piece_type, chess.WHITE)) * value
        black_total += len(board.pieces(piece_type, chess.BLACK)) * value
    return white_total, black_total, white_total - black_total


def board_feature_snapshot(
    board: chess.Board,
    *,
    eval_cp_stm: int | None = None,
    eval_mate_stm: int | None = None,
) -> dict[str, float | int | str]:
    w_total, b_total, balance = _material_snapshot(board)
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)

    return {
        "turn": "w" if board.turn == chess.WHITE else "b",
        "material_white": float(w_total),
        "material_black": float(b_total),
        "material_balance": float(balance),
        "mobility": int(board.legal_moves.count()),
        "is_check": int(board.is_check()),
        "is_checkmate": int(board.is_checkmate()),
        "is_stalemate": int(board.is_stalemate()),
        "white_king_file": -1 if w_king is None else chess.square_file(w_king),
        "white_king_rank": -1 if w_king is None else chess.square_rank(w_king),
        "black_king_file": -1 if b_king is None else chess.square_file(b_king),
        "black_king_rank": -1 if b_king is None else chess.square_rank(b_king),
        "eval_cp_stm": 0 if eval_cp_stm is None else int(eval_cp_stm),
        "eval_norm_stm": float(normalize_eval_for_embedding(cp=eval_cp_stm, mate=eval_mate_stm)),
    }


def snapshot_delta_rows(
    query_snapshot: dict[str, float | int | str],
    candidate_snapshot: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    fields = [
        "material_white",
        "material_black",
        "material_balance",
        "mobility",
        "is_check",
        "is_checkmate",
        "is_stalemate",
        "white_king_file",
        "white_king_rank",
        "black_king_file",
        "black_king_rank",
        "eval_cp_stm",
        "eval_norm_stm",
    ]
    rows: list[dict[str, Any]] = []
    for field in fields:
        qv = query_snapshot.get(field)
        cv = candidate_snapshot.get(field)
        qf = float(qv) if isinstance(qv, (int, float)) else 0.0
        cf = float(cv) if isinstance(cv, (int, float)) else 0.0
        rows.append({"feature": field, "query": qf, "candidate": cf, "delta": cf - qf})
    return rows


def explanation_chips(
    pair_explain: dict[str, Any],
    query_snapshot: dict[str, float | int | str],
    candidate_snapshot: dict[str, float | int | str],
    *,
    limit: int = 4,
) -> list[str]:
    chips: list[str] = []

    def _add(label: str) -> None:
        if label not in chips and len(chips) < max(1, int(limit)):
            chips.append(label)

    material_delta = abs(float(candidate_snapshot.get("material_balance", 0.0)) - float(query_snapshot.get("material_balance", 0.0)))
    mobility_delta = abs(float(candidate_snapshot.get("mobility", 0.0)) - float(query_snapshot.get("mobility", 0.0)))
    eval_delta = abs(float(candidate_snapshot.get("eval_norm_stm", 0.0)) - float(query_snapshot.get("eval_norm_stm", 0.0)))
    king_delta = (
        abs(float(candidate_snapshot.get("white_king_file", 0.0)) - float(query_snapshot.get("white_king_file", 0.0)))
        + abs(float(candidate_snapshot.get("white_king_rank", 0.0)) - float(query_snapshot.get("white_king_rank", 0.0)))
        + abs(float(candidate_snapshot.get("black_king_file", 0.0)) - float(query_snapshot.get("black_king_file", 0.0)))
        + abs(float(candidate_snapshot.get("black_king_rank", 0.0)) - float(query_snapshot.get("black_king_rank", 0.0)))
    )

    contributions = sorted(
        pair_explain.get("group_contributions", []),
        key=lambda row: abs(float(row.get("share_abs_pct", 0.0))),
        reverse=True,
    )
    top_groups = [str(row.get("group", "")) for row in contributions[:3]]

    if "base_781" in top_groups:
        _add("Structure match")
    if material_delta <= 0.5:
        _add("Same material")
    elif material_delta >= 2.0:
        _add("Material differs")
    if mobility_delta <= 3.0:
        _add("Similar mobility")
    elif mobility_delta >= 8.0:
        _add("Mobility differs")
    if king_delta <= 2.0:
        _add("Similar king placement")
    elif king_delta >= 5.0:
        _add("King setup differs")
    if eval_delta >= 0.12:
        _add("Eval differs")
    elif "eval" in top_groups:
        _add("Eval aligned")
    if int(candidate_snapshot.get("is_check", 0)) != int(query_snapshot.get("is_check", 0)):
        _add("Check state differs")

    for group in top_groups:
        if group == "piece_counts":
            _add("Piece-count match")
        elif group == "piece_info":
            _add("Tactical state match")
        elif group == "material":
            _add("Material-driven")

    return chips[: max(1, int(limit))]


def similarity_narrative(
    *,
    score: float,
    band: str,
    pair_explain: dict[str, Any],
    query_snapshot: dict[str, float | int | str],
    candidate_snapshot: dict[str, float | int | str],
) -> str:
    chips = explanation_chips(pair_explain, query_snapshot, candidate_snapshot, limit=3)
    contrib_rows = sorted(
        pair_explain.get("group_contributions", []),
        key=lambda row: abs(float(row.get("share_abs_pct", 0.0))),
        reverse=True,
    )
    group_phrases = {
        "base_781": "core board structure",
        "piece_counts": "piece-count profile",
        "piece_info": "king safety and tactical state",
        "material": "material balance",
        "eval": "engine-evaluation context",
    }
    driver_text = ", ".join(group_phrases.get(str(row.get("group")), str(row.get("group"))) for row in contrib_rows[:2]) or "the active features"

    material_delta = float(candidate_snapshot.get("material_balance", 0.0)) - float(query_snapshot.get("material_balance", 0.0))
    mobility_delta = float(candidate_snapshot.get("mobility", 0.0)) - float(query_snapshot.get("mobility", 0.0))
    eval_delta = float(candidate_snapshot.get("eval_norm_stm", 0.0)) - float(query_snapshot.get("eval_norm_stm", 0.0))

    details: list[str] = []
    if abs(material_delta) >= 0.5:
        details.append(f"material shifts by {material_delta:+.1f}")
    if abs(mobility_delta) >= 3.0:
        details.append(f"mobility shifts by {mobility_delta:+.0f} moves")
    if abs(eval_delta) >= 0.12:
        details.append(f"the eval feature moves by {eval_delta:+.2f}")

    tail = "; ".join(details) if details else "the higher-level deltas stay fairly close"
    chip_text = ", ".join(chips) if chips else "feature overlap"
    return (
        f"This match lands in the {band.upper()} band at {score:.1f}/100. "
        f"It is mostly driven by {driver_text}, with {chip_text.lower()}; {tail}."
    )


def explain_pair(
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    *,
    group_slices: dict[str, slice],
) -> dict[str, Any]:
    similarity, distance = cosine_similarity_distance(query_vec, candidate_vec)
    contributions = group_contributions_cosine(query_vec, candidate_vec, group_slices)
    return {
        "similarity": float(similarity),
        "distance": float(distance),
        "score_100": score_from_distance(distance),
        "group_contributions": contributions,
    }
