from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import chess


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "demo_entry"


def _load_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def upsert_demo_position(
    positions_path: str | Path,
    *,
    position_id: str,
    name: str,
    fen: str,
    description: str = "",
) -> dict[str, Any]:
    board = chess.Board(fen)
    payload = _load_json(positions_path, {"positions": []})
    rows = payload.get("positions")
    if not isinstance(rows, list):
        rows = []
        payload["positions"] = rows

    position_id = _slugify(position_id or name)
    entry = {
        "id": position_id,
        "name": name.strip() or position_id,
        "fen": board.fen(en_passant="fen"),
        "description": description.strip(),
    }

    for idx, row in enumerate(rows):
        if isinstance(row, dict) and str(row.get("id", "")) == position_id:
            rows[idx] = entry
            _write_json(positions_path, payload)
            return entry

    rows.append(entry)
    _write_json(positions_path, payload)
    return entry


def upsert_demo_scenario(
    scenarios_path: str | Path,
    *,
    scenario_id: str,
    title: str,
    query_ref: dict[str, Any],
    candidate_ref: dict[str, Any],
    expected_band: str,
    feature_focus: list[str] | None = None,
    narrative: str = "",
    comparison_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _load_json(
        scenarios_path,
        {
            "version": 1,
            "default_method": "brute",
            "default_score_bands": {
                "perfect": [95.0, 100.0],
                "high": [80.0, 94.99],
                "medium": [60.0, 79.99],
                "low": [0.0, 59.99],
            },
            "scenarios": [],
        },
    )
    rows = payload.get("scenarios")
    if not isinstance(rows, list):
        rows = []
        payload["scenarios"] = rows

    scenario_id = _slugify(scenario_id or title)
    entry = {
        "id": scenario_id,
        "title": title.strip() or scenario_id,
        "query": dict(query_ref),
        "candidate": dict(candidate_ref),
        "expected_band": expected_band.strip().lower(),
        "feature_focus": list(feature_focus or []),
        "narrative": narrative.strip(),
    }
    if comparison_candidates:
        entry["comparison_candidates"] = [dict(item) for item in comparison_candidates]

    for idx, row in enumerate(rows):
        if isinstance(row, dict) and str(row.get("id", "")) == scenario_id:
            rows[idx] = entry
            _write_json(scenarios_path, payload)
            return entry

    rows.append(entry)
    _write_json(scenarios_path, payload)
    return entry
