from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any

import chess
import chess.svg
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from benchmarks import run_search_benchmark
from chess_state import AnalysisTree
from components.cg_board import cg_board
from demo_authoring import upsert_demo_position, upsert_demo_scenario
from encoder import EncoderConfig, PositionEncoder, validate_fen
from explainability import (
    board_feature_snapshot,
    default_score_bands,
    explanation_chips,
    explain_pair,
    normalize_score_bands,
    score_band,
    score_from_distance,
    similarity_narrative,
    snapshot_delta_rows,
)
from eval_engine import (
    DEFAULT_STOCKFISH_PATH,
    EngineEval,
    analyze_position,
    eval_bar_fraction_from_normalized_white,
    format_eval_numeric,
    normalize_eval_for_embedding,
)
from pgn_loader import (
    CorpusBundle,
    GameRecord,
    build_position_corpus_in_memory,
    game_board_at_ply,
    list_pgn_files,
    list_sample_pgn_files,
    load_games_from_paths,
    load_curated_sample_games,
)
from search import available_methods, build_index, faiss_available, hits_with_metadata


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
SAMPLE_GAMES_DIR = DATA_DIR / "sample_games"
PRESETS_PATH = DATA_DIR / "presets.json"
DEFAULT_BULK_PGN_DIR = DATA_DIR / "bulk_pgn"
DEMO_DIR = DATA_DIR / "demo"
DEMO_GAMES_PATH = DEMO_DIR / "demo_games.pgn"
DEMO_SCENARIOS_PATH = DEMO_DIR / "demo_scenarios.json"
DEMO_POSITIONS_PATH = DEMO_DIR / "demo_positions.json"
MAIN_BOARD_HEIGHT_PX = 560
PUBLIC_APP_MODE = True
PUBLIC_DEFAULT_ENGINE_DEPTH = 8
PUBLIC_MAX_ENGINE_DEPTH = 12
PUBLIC_DEFAULT_CORPUS_MAX_POSITIONS = 1500
PUBLIC_MAX_CORPUS_POSITIONS = 10000
PUBLIC_DEFAULT_BENCH_QUERY_COUNT = 20
PUBLIC_MAX_BENCH_QUERY_COUNT = 75


def resolved_default_stockfish_path() -> str:
    is_windows = sys.platform.startswith("win")
    local_candidates = [
        APP_ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        APP_ROOT / "stockfish" / "stockfish.exe",
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
    ]
    for candidate in local_candidates:
        if candidate.suffix.lower() == ".exe" and not is_windows:
            continue
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    which_path = shutil.which("stockfish")
    if which_path:
        return str(which_path)
    return DEFAULT_STOCKFISH_PATH


st.set_page_config(page_title="Chess Similarity at Scale Demo", page_icon="CS", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #1f1a14;
          --muted: #64584c;
          --panel: rgba(255, 252, 246, 0.88);
          --line: rgba(62, 47, 33, 0.18);
          --accent: #0f766e;
          --accent-2: #9a3412;
        }
        .stApp {
          background:
            radial-gradient(1200px 420px at 10% -10%, rgba(15,118,110,0.14), transparent 55%),
            radial-gradient(900px 320px at 100% 0%, rgba(180,83,9,0.12), transparent 50%),
            linear-gradient(180deg, #fbf8f3 0%, #f3eee4 100%);
          font-family: "Segoe UI", "Calibri", "Helvetica Neue", sans-serif;
        }
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
          color: var(--ink);
        }
        [data-testid="stMarkdownContainer"] * {
          color: var(--ink) !important;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
          color: var(--ink) !important;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
          color: var(--ink);
        }
        .stApp a {
          color: #0b5f57;
        }
        .block-container { padding-top: 1.05rem; padding-bottom: 1.1rem; }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(250,245,235,0.98), rgba(243,235,223,0.96));
          border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
          color: var(--ink) !important;
        }
        [data-testid="stSidebar"] .stCaption, .stCaption {
          color: var(--muted) !important;
        }
        [data-baseweb="input"] input, [data-baseweb="input"] textarea {
          color: var(--ink) !important;
          background: rgba(255, 255, 255, 0.92) !important;
          border-color: rgba(88, 69, 52, 0.3) !important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {
          color: var(--ink) !important;
          background: rgba(255,255,255,0.95) !important;
          -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stTextArea"] textarea {
          font-family: "Cascadia Code", "Consolas", monospace;
          border-color: rgba(88, 69, 52, 0.3) !important;
        }
        [data-baseweb="select"] * {
          color: var(--ink) !important;
        }
        [data-baseweb="select"] > div {
          color: var(--ink) !important;
          background: rgba(255,255,255,0.94) !important;
          border-color: rgba(88, 69, 52, 0.3) !important;
        }
        [data-baseweb="slider"] * {
          color: var(--ink) !important;
        }
        button[data-baseweb="tab"] {
          color: var(--ink) !important;
          background: rgba(255,255,255,0.46);
          border: 1px solid rgba(88, 69, 52, 0.2);
          border-radius: 8px 8px 0 0;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
          background: rgba(232, 246, 243, 0.9);
          border-color: rgba(15,118,110,0.3);
        }
        [data-testid="stAlert"] * {
          color: var(--ink) !important;
        }
        [data-testid="stExpander"] details {
          background: rgba(255,255,255,0.52);
          border: 1px solid rgba(88,69,52,0.18);
          border-radius: 10px;
          padding: 0.2rem 0.35rem;
        }
        .stButton > button {
          border: 1px solid rgba(67, 50, 32, 0.28);
          color: var(--ink) !important;
          background: linear-gradient(180deg, #fffaf3, #f2e7d7);
          border-radius: 10px;
          box-shadow: 0 2px 10px rgba(40, 30, 20, 0.08);
        }
        .stButton > button:hover {
          border-color: rgba(15,118,110,0.45);
          box-shadow: 0 4px 14px rgba(15,118,110,0.15);
          transform: translateY(-1px);
        }
        .stButton > button[kind="primary"] {
          background: linear-gradient(180deg, #d8f1ed, #c7e8e2);
          border-color: rgba(15,118,110,0.4);
        }
        [data-testid="stCaptionContainer"] code,
        [data-testid="stMarkdownContainer"] code,
        [data-testid="stMarkdownContainer"] pre {
          color: #20190f !important;
          background: rgba(248, 242, 232, 0.92) !important;
          border: 1px solid rgba(88,69,52,0.2) !important;
          border-radius: 6px !important;
        }
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCode"] pre,
        .stCode pre {
          color: #20190f !important;
          background: rgba(248, 242, 232, 0.95) !important;
          border: 1px solid rgba(88,69,52,0.2) !important;
          border-radius: 10px !important;
          font-family: "Cascadia Code", "Consolas", monospace !important;
        }
        [data-testid="stJson"] {
          background: rgba(255, 255, 255, 0.78) !important;
          border: 1px solid rgba(88, 69, 52, 0.24) !important;
          border-radius: 10px !important;
          padding: 0.3rem 0.45rem !important;
        }
        [data-testid="stJson"] * {
          color: #20190f !important;
          -webkit-text-fill-color: #20190f !important;
        }
        [data-testid="stDataFrame"] * {
          color: var(--ink) !important;
        }
        .cap-hero {
          border: 1px solid rgba(60, 47, 34, 0.12); border-radius: 18px;
          background: linear-gradient(140deg, rgba(255,255,255,0.82), rgba(247,242,233,0.92));
          padding: 0.9rem 1rem; box-shadow: 0 10px 26px rgba(45,34,23,0.08); margin-bottom: 0.75rem;
        }
        .cap-hero h1 { margin: 0; font-size: 1.25rem; color: #231c14; letter-spacing: 0.02em; }
        .cap-hero p { margin: 0.25rem 0 0; color: #6e6457; font-size: 0.92rem; }
        .cap-card {
          border: 1px solid rgba(60,47,34,0.14); border-radius: 16px; background: var(--panel);
          box-shadow: 0 8px 22px rgba(42,33,23,0.05); padding: 0.75rem 0.9rem; margin-bottom: 0.7rem;
          backdrop-filter: blur(3px);
        }
        .cap-card * {
          color: var(--ink) !important;
        }
        .cap-card .stCaption {
          color: var(--muted) !important;
        }
        .cap-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 700; }
        .cap-value { color: var(--ink); font-weight: 700; font-size: 1.05rem; }
        .eval-bar-wrap {
          height: 560px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(52,44,33,0.18);
          background: #1e1a16; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
          box-sizing: border-box;
          margin: 0;
        }
        .eval-bar-black { width: 100%; background: #1d1a17; }
        .eval-bar-white { width: 100%; background: linear-gradient(180deg, #fffaf2, #efe7d8); }
        .branch-pill {
          display: inline-block; border: 1px solid rgba(15,118,110,0.2); background: rgba(15,118,110,0.08);
          color: #0f766e; padding: 0.15rem 0.45rem; border-radius: 999px; font-size: 0.76rem;
          margin-right: 0.25rem; margin-bottom: 0.25rem;
        }
        .similar-result-row {
          border: 1px solid rgba(80, 62, 43, 0.14);
          border-radius: 12px;
          padding: 0.55rem 0.6rem 0.4rem 0.6rem;
          margin-bottom: 0.55rem;
          background: rgba(255,255,255,0.78);
        }
        .mini-board-shell {
          border: 1px solid rgba(78,60,40,0.24);
          border-radius: 8px;
          overflow: hidden;
          background: #f3ead8;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.5);
        }
        .reason-chip {
          display: inline-block;
          padding: 0.18rem 0.52rem;
          margin: 0 0.35rem 0.35rem 0;
          border-radius: 999px;
          border: 1px solid rgba(15,118,110,0.18);
          background: rgba(15,118,110,0.08);
          color: #0f5d58;
          font-size: 0.76rem;
          font-weight: 600;
        }
        .compare-panel {
          border: 1px solid rgba(60,47,34,0.14);
          border-radius: 16px;
          background: rgba(255,255,255,0.72);
          padding: 0.75rem 0.85rem;
          margin-top: 0.8rem;
        }
        .compare-note {
          color: var(--muted);
          font-size: 0.86rem;
        }
        .square-pill {
          display: inline-block;
          padding: 0.14rem 0.42rem;
          margin: 0 0.28rem 0.28rem 0;
          border-radius: 999px;
          border: 1px solid rgba(14, 116, 144, 0.26);
          background: rgba(56, 189, 248, 0.14);
          color: #0c4a6e;
          font-size: 0.76rem;
          font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_presets_cached(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    presets: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fen = str(item.get("fen", "")).strip()
        ok, _ = validate_fen(fen)
        if not ok:
            continue
        presets.append(
            {
                "name": str(item.get("name", "Preset")),
                "fen": fen,
                "description": str(item.get("description", "")),
            }
        )
    return presets


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def eval_override_from_ref_meta(meta: dict[str, Any] | None) -> tuple[int | None, int | None] | None:
    if not isinstance(meta, dict):
        return None
    cp = _as_optional_int(meta.get("eval_cp_stm"))
    mate = _as_optional_int(meta.get("eval_mate_stm"))
    if cp is None and mate is None:
        return None
    return cp, mate


def method_options_for_ui() -> list[str]:
    if faiss_available():
        return available_methods()
    return ["brute"]


@st.cache_data(show_spinner=False)
def load_sample_games_cached(sample_dir: str) -> dict[str, GameRecord]:
    return load_curated_sample_games(sample_dir)


@st.cache_data(show_spinner=False)
def list_pgn_paths_cached(folder: str, recursive: bool) -> list[str]:
    return [str(p) for p in list_pgn_files(folder, recursive=recursive)]


@st.cache_data(show_spinner=False)
def load_demo_positions_cached(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos_id = str(item.get("id", "")).strip()
        fen = str(item.get("fen", "")).strip()
        if not pos_id:
            continue
        ok, _ = validate_fen(fen)
        if not ok:
            continue
        out.append(
            {
                "id": pos_id,
                "name": str(item.get("name", pos_id)),
                "fen": fen,
                "description": str(item.get("description", "")),
                "eval_cp_stm": _as_optional_int(item.get("eval_cp_stm")),
                "eval_mate_stm": _as_optional_int(item.get("eval_mate_stm")),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def load_demo_scenarios_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    defaults = {
        "version": 1,
        "default_method": "brute",
        "default_top_k": 6,
        "default_score_bands": default_score_bands(),
        "scenarios": [],
    }
    if not path.exists():
        return defaults
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        return defaults

    scenarios = raw.get("scenarios", [])
    clean_scenarios = [s for s in scenarios if isinstance(s, dict)]
    return {
        "version": int(raw.get("version", 1)),
        "default_method": str(raw.get("default_method", "brute")),
        "default_top_k": int(raw.get("default_top_k", 6)),
        "default_score_bands": normalize_score_bands(raw.get("default_score_bands")),
        "scenarios": clean_scenarios,
    }


def demo_profile_configs() -> dict[str, EncoderConfig]:
    return {
        "A: base_781": EncoderConfig(
            include_eval_score=False,
            include_piece_counts=False,
            include_piece_info=False,
            include_material_balance=False,
        ),
        "B: base+domain": EncoderConfig(
            include_eval_score=False,
            include_piece_counts=True,
            include_piece_info=True,
            include_material_balance=True,
        ),
        "C: full+eval": EncoderConfig(
            include_eval_score=True,
            include_piece_counts=True,
            include_piece_info=True,
            include_material_balance=True,
        ),
    }


@st.cache_data(show_spinner=False)
def build_demo_dataset_cached(
    demo_games_path_str: str,
    demo_positions_path_str: str,
    encoder_cfg_json: str,
    *,
    stockfish_path: str,
    eval_depth: int,
    use_engine_eval_for_eval_feature: bool,
) -> dict[str, Any]:
    cfg = EncoderConfig.from_dict(json.loads(encoder_cfg_json))
    encoder = PositionEncoder(cfg)
    demo_positions = load_demo_positions_cached(demo_positions_path_str)
    positions_by_id = {p["id"]: p for p in demo_positions}

    eval_provider = None
    if cfg.include_eval_score and use_engine_eval_for_eval_feature and stockfish_path.strip():

        def _eval_provider(board: chess.Board) -> tuple[int | None, int | None] | None:
            e = analyze_position(board, stockfish_path=stockfish_path, depth=eval_depth)
            if not e.ok:
                return (None, None)
            return (e.cp_stm, e.mate_stm)

        eval_provider = _eval_provider

    demo_games_path = Path(demo_games_path_str)
    games_by_uid: dict[str, GameRecord] = {}
    vectors: list[np.ndarray] = []
    positions: list[dict[str, Any]] = []

    if demo_games_path.exists():
        games_by_uid = load_games_from_paths([demo_games_path])
        game_bundle = build_position_corpus_in_memory(
            [demo_games_path],
            encoder,
            every_n_plies=1,
            min_ply=0,
            dedupe_games=False,
            dedupe_fen=False,
            include_start_position=True,
            eval_provider=eval_provider,
        )
        vectors.extend([row for row in game_bundle.vectors])
        positions.extend(game_bundle.positions)
        games_by_uid.update(game_bundle.games_by_uid)

    next_index = len(positions)
    for entry in demo_positions:
        board = chess.Board(entry["fen"])
        cp = _as_optional_int(entry.get("eval_cp_stm"))
        mate = _as_optional_int(entry.get("eval_mate_stm"))
        if cp is None and mate is None and eval_provider is not None:
            eval_pair = eval_provider(board)
            if eval_pair is not None:
                cp, mate = eval_pair
        vec = encoder.encode(board, eval_cp=cp, eval_mate=mate)
        vectors.append(vec)
        demo_game_id = f"demo_pos::{entry['id']}"
        positions.append(
            {
                "corpus_index": next_index,
                "source_pgn_file": "demo_positions",
                "source_path": str(Path(demo_positions_path_str)),
                "game_id": demo_game_id,
                "game_index": -1,
                "ply_index": 0,
                "move_index": -1,
                "fullmove_number": board.fullmove_number,
                "turn": "w" if board.turn == chess.WHITE else "b",
                "fen": board.fen(en_passant="fen"),
                "result": "*",
                "headers": {"Event": entry["name"], "White": "Demo", "Black": "Demo"},
                "position_id": entry["id"],
                "position_name": entry["name"],
                "eval_cp_stm": cp,
                "eval_mate_stm": mate,
            }
        )
        games_by_uid[demo_game_id] = GameRecord(
            game_uid=demo_game_id,
            source_pgn_file="demo_positions",
            source_path=str(Path(demo_positions_path_str)),
            game_index=-1,
            headers={"Event": entry["name"], "White": "Demo", "Black": "Demo", "Result": "*"},
            start_fen=board.fen(en_passant="fen"),
            moves_uci=[],
            moves_san=[],
            result="*",
        )
        next_index += 1

    matrix = np.vstack(vectors).astype(np.float32, copy=False) if vectors else np.zeros((0, encoder.dim), dtype=np.float32)
    return {
        "vectors": matrix,
        "positions": positions,
        "games_by_uid": games_by_uid,
        "positions_by_id": positions_by_id,
        "manifest": {
            "demo_games_path": str(demo_games_path),
            "demo_positions_count": len(demo_positions),
            "positions_total": len(positions),
            "encoder": cfg.to_dict(),
            "encoder_dim": encoder.dim,
        },
    }


def resolve_demo_ref(
    ref: dict[str, Any],
    *,
    positions_by_id: dict[str, dict[str, Any]],
    games_by_uid: dict[str, GameRecord],
) -> tuple[str, str, dict[str, Any]]:
    if "fen" in ref:
        fen = str(ref.get("fen", "")).strip()
        ok, err = validate_fen(fen)
        if not ok:
            raise ValueError(f"Invalid FEN in scenario ref: {err}")
        return fen, "Direct FEN", {"source": "fen", "eval_cp_stm": _as_optional_int(ref.get("eval_cp_stm")), "eval_mate_stm": _as_optional_int(ref.get("eval_mate_stm"))}

    position_id = str(ref.get("position_id", "")).strip()
    if position_id:
        entry = positions_by_id.get(position_id)
        if entry is None:
            raise KeyError(f"Unknown demo position id: {position_id}")
        return (
            str(entry["fen"]),
            str(entry.get("name", position_id)),
            {
                "source": "position_id",
                "position_id": position_id,
                "eval_cp_stm": _as_optional_int(entry.get("eval_cp_stm")),
                "eval_mate_stm": _as_optional_int(entry.get("eval_mate_stm")),
            },
        )

    game_id = str(ref.get("game_id", "")).strip()
    if game_id:
        if game_id not in games_by_uid:
            raise KeyError(f"Unknown game id in scenario ref: {game_id}")
        ply_index = int(ref.get("ply_index", 0))
        game = games_by_uid[game_id]
        board = game_board_at_ply(game, ply_index)
        label = f"{game.title()} @ ply {ply_index}"
        return (
            board.fen(en_passant="fen"),
            label,
            {
                "source": "game_ref",
                "game_id": game_id,
                "ply_index": ply_index,
                "eval_cp_stm": _as_optional_int(ref.get("eval_cp_stm")),
                "eval_mate_stm": _as_optional_int(ref.get("eval_mate_stm")),
            },
        )

    raise ValueError("Scenario ref must include one of: fen, position_id, game_id.")


@st.cache_data(show_spinner=False)
def build_corpus_from_paths_cached(
    pgn_paths_json: str,
    encoder_cfg_json: str,
    *,
    every_n_plies: int,
    min_ply: int,
    max_positions: int,
    dedupe_games: bool,
    dedupe_fen: bool,
    use_engine_eval_for_corpus: bool,
    stockfish_path: str,
    eval_depth: int,
) -> CorpusBundle:
    cfg = EncoderConfig.from_dict(json.loads(encoder_cfg_json))
    encoder = PositionEncoder(cfg)
    pgn_paths = [Path(p) for p in json.loads(pgn_paths_json)]

    eval_provider = None
    if cfg.include_eval_score and use_engine_eval_for_corpus and stockfish_path.strip():
        def _eval_provider(board: chess.Board) -> tuple[int | None, int | None] | None:
            e = analyze_position(board, stockfish_path=stockfish_path, depth=eval_depth)
            if not e.ok:
                return (None, None)
            return (e.cp_stm, e.mate_stm)

        eval_provider = _eval_provider

    return build_position_corpus_in_memory(
        pgn_paths,
        encoder,
        every_n_plies=max(1, every_n_plies),
        min_ply=max(0, min_ply),
        max_positions=max_positions if max_positions > 0 else None,
        dedupe_games=dedupe_games,
        dedupe_fen=dedupe_fen,
        eval_provider=eval_provider,
    )


@st.cache_data(show_spinner=False)
def build_corpus_cached(
    sample_dir: str,
    encoder_cfg_json: str,
    *,
    every_n_plies: int,
    min_ply: int,
    max_positions: int,
    dedupe_games: bool,
    dedupe_fen: bool,
    use_engine_eval_for_corpus: bool,
    stockfish_path: str,
    eval_depth: int,
) -> CorpusBundle:
    pgn_paths = [str(p) for p in list_sample_pgn_files(sample_dir)]
    return build_corpus_from_paths_cached(
        json.dumps(pgn_paths, sort_keys=True),
        encoder_cfg_json,
        every_n_plies=every_n_plies,
        min_ply=min_ply,
        max_positions=max_positions,
        dedupe_games=dedupe_games,
        dedupe_fen=dedupe_fen,
        use_engine_eval_for_corpus=use_engine_eval_for_corpus,
        stockfish_path=stockfish_path,
        eval_depth=eval_depth,
    )


@st.cache_resource(show_spinner=False)
def build_search_index_cached(method: str, metric: str, vectors, config_json: str):
    cfg = json.loads(config_json) if config_json.strip() else {}
    return build_index(method, vectors, metric=metric, config=cfg)


def ensure_session_state() -> None:
    if "main_tree" not in st.session_state:
        st.session_state.main_tree = AnalysisTree()
    if "similar_tree" not in st.session_state:
        st.session_state.similar_tree = None
    if "similar_selected_meta" not in st.session_state:
        st.session_state.similar_selected_meta = None
    if "main_board_instance_id" not in st.session_state:
        st.session_state.main_board_instance_id = 0
    if "similar_board_instance_id" not in st.session_state:
        st.session_state.similar_board_instance_id = 0
    if "main_last_event_signature" not in st.session_state:
        st.session_state.main_last_event_signature = None
    if "ui_notice" not in st.session_state:
        st.session_state.ui_notice = None
    if "last_benchmark_report" not in st.session_state:
        st.session_state.last_benchmark_report = None
    if "fen_input_text" not in st.session_state:
        st.session_state.fen_input_text = st.session_state.main_tree.current_fen
    if "pending_fen_input_text" not in st.session_state:
        st.session_state.pending_fen_input_text = None
    if "main_restore_stack" not in st.session_state:
        st.session_state.main_restore_stack = []
    if "main_board_render_px" not in st.session_state:
        st.session_state.main_board_render_px = MAIN_BOARD_HEIGHT_PX
    if "main_game_id" not in st.session_state:
        st.session_state.main_game_id = None
    if "similar_game_id" not in st.session_state:
        st.session_state.similar_game_id = None
    if "compare_mode_active" not in st.session_state:
        st.session_state.compare_mode_active = False


def set_notice(level: str, message: str) -> None:
    st.session_state.ui_notice = (level, message)


def show_notice_once() -> None:
    notice = st.session_state.get("ui_notice")
    if not notice:
        return
    st.session_state.ui_notice = None
    level, message = notice
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def bump_main_board_instance() -> None:
    st.session_state.main_board_instance_id += 1
    st.session_state.main_last_event_signature = None


def bump_similar_board_instance() -> None:
    st.session_state.similar_board_instance_id += 1


def sync_fen_input_to_main() -> None:
    # Defer assignment until before the text_input widget is instantiated.
    st.session_state.pending_fen_input_text = st.session_state.main_tree.current_fen


def _tree_context_label(tree: AnalysisTree) -> str:
    event = tree.headers.get("Event", "Analysis")
    return f"{event} @ ply {tree.current_ply_index}"


def clear_main_restore_stack() -> None:
    st.session_state.main_restore_stack = []


def push_main_snapshot(reason: str) -> None:
    stack = st.session_state.main_restore_stack
    stack.append(
        {
            "reason": reason,
            "label": _tree_context_label(st.session_state.main_tree),
            "tree": st.session_state.main_tree.clone(),
            "main_game_id": st.session_state.get("main_game_id"),
        }
    )
    if len(stack) > 10:
        del stack[0]


def restore_previous_main_snapshot() -> bool:
    stack = st.session_state.get("main_restore_stack", [])
    if not stack:
        return False
    snap = stack.pop()
    tree = snap.get("tree")
    if not isinstance(tree, AnalysisTree):
        return False
    st.session_state.main_tree = tree
    st.session_state.main_game_id = snap.get("main_game_id")
    bump_main_board_instance()
    sync_fen_input_to_main()
    set_notice("success", f"Restored previous main game context: {snap.get('label', 'snapshot')}")
    return True


def load_main_tree_from_game(game: GameRecord, *, current_ply: int = 0, clear_restore_stack_flag: bool = False) -> None:
    st.session_state.main_tree = AnalysisTree.from_mainline(
        start_fen=game.start_fen, moves_uci=game.moves_uci, headers=game.headers, current_ply=current_ply
    )
    st.session_state.main_game_id = game.game_uid
    if clear_restore_stack_flag:
        clear_main_restore_stack()
    bump_main_board_instance()
    sync_fen_input_to_main()


def load_similar_tree_from_game(game: GameRecord, *, current_ply: int = 0) -> None:
    st.session_state.similar_tree = AnalysisTree.from_mainline(
        start_fen=game.start_fen, moves_uci=game.moves_uci, headers=game.headers, current_ply=current_ply
    )
    st.session_state.similar_game_id = game.game_uid
    bump_similar_board_instance()


def sync_similar_into_main() -> None:
    tree = st.session_state.get("similar_tree")
    if tree is None:
        return
    st.session_state.main_tree = tree.clone()
    st.session_state.main_game_id = st.session_state.get("similar_game_id")
    bump_main_board_instance()
    sync_fen_input_to_main()


def render_eval_bar(eval_info: EngineEval, *, height_px: int = MAIN_BOARD_HEIGHT_PX) -> None:
    frac_white = eval_info.eval_bar_white if eval_info.ok else 0.5
    frac_white = max(0.0, min(1.0, frac_white))
    black_pct = (1.0 - frac_white) * 100.0
    white_pct = frac_white * 100.0
    st.markdown(
        f"""
        <div class="eval-bar-wrap" style="height:{int(height_px)}px">
          <div class="eval-bar-black" style="height:{black_pct:.2f}%"></div>
          <div class="eval-bar-white" style="height:{white_pct:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_board_thumbnail(fen: str, *, orientation: str, size: int = 188) -> None:
    try:
        board = chess.Board(fen)
        orient = chess.WHITE if orientation == "white" else chess.BLACK
        svg = chess.svg.board(
            board=board,
            size=size,
            coordinates=False,
            orientation=orient,
        )
        html = f'<div class="mini-board-shell">{svg}</div>'
    except Exception:
        html = (
            f'<div class="mini-board-shell" style="height:{int(size)}px;display:flex;align-items:center;'
            f'justify-content:center;">Invalid FEN</div>'
        )
    st.components.v1.html(html, height=size + 12, scrolling=False)


def render_chip_row(chips: list[str]) -> None:
    if not chips:
        return
    html = "".join(f'<span class="reason-chip">{chip}</span>' for chip in chips)
    st.markdown(html, unsafe_allow_html=True)


def board_difference_squares(left_board: chess.Board, right_board: chess.Board) -> list[str]:
    all_squares = sorted(set(left_board.piece_map()) | set(right_board.piece_map()))
    changed = [
        chess.square_name(square)
        for square in all_squares
        if left_board.piece_at(square) != right_board.piece_at(square)
    ]
    return changed


def compare_state_notes(left_board: chess.Board, right_board: chess.Board) -> list[str]:
    notes: list[str] = []
    if left_board.turn != right_board.turn:
        notes.append("Side to move differs")
    if left_board.castling_rights != right_board.castling_rights:
        notes.append("Castling rights differ")
    if left_board.ep_square != right_board.ep_square:
        notes.append("En-passant state differs")
    return notes


def render_large_compare_board(
    board: chess.Board,
    *,
    orientation: str,
    size: int,
    highlight_squares: list[str] | None = None,
) -> None:
    orient = chess.WHITE if orientation == "white" else chess.BLACK
    fills = {
        chess.parse_square(square_name): "#38bdf8"
        for square_name in (highlight_squares or [])
    }
    svg = chess.svg.board(
        board=board,
        size=size,
        orientation=orient,
        coordinates=True,
        fill=fills or None,
    )
    st.components.v1.html(f'<div class="mini-board-shell">{svg}</div>', height=size + 16, scrolling=False)


def styled_benchmark_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(background="white")
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelFontSize=14,
            titleFontSize=16,
            labelColor="#2b2b2b",
            titleColor="#1f1f1f",
            gridColor="#e8e8e8",
            tickColor="#cfcfcf",
        )
        .configure_legend(
            titleFontSize=14,
            labelFontSize=13,
            titleColor="#1f1f1f",
            labelColor="#2b2b2b",
            orient="right",
        )
        .configure_title(fontSize=18, color="#1f1f1f")
    )


def build_similarity_english_summary(
    *,
    score: float,
    band: str,
    pair_explain: dict[str, Any],
    query_snapshot: dict[str, float | int | str],
    candidate_snapshot: dict[str, float | int | str],
) -> str:
    return similarity_narrative(
        score=score,
        band=band,
        pair_explain=pair_explain,
        query_snapshot=query_snapshot,
        candidate_snapshot=candidate_snapshot,
    )


def tree_nav_controls(tree: AnalysisTree, *, key_prefix: str, title: str) -> bool:
    changed = False
    st.markdown(f"**{title}**")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start", key=f"{key_prefix}_start", use_container_width=True):
        tree.jump_to_start()
        changed = True
    if c2.button("Back", key=f"{key_prefix}_back", use_container_width=True):
        changed = tree.step_back() or changed
    if c3.button("Forward", key=f"{key_prefix}_fwd", use_container_width=True):
        changed = tree.step_forward() or changed
    if c4.button("End", key=f"{key_prefix}_end", use_container_width=True):
        tree.jump_to_end()
        changed = True

    st.caption(f"Current ply: {tree.current_ply_index} | Variations here: {len(tree.current.children)}")

    siblings = tree.sibling_variations()
    if len(siblings) > 1:
        st.caption("Switch branch at current ply:")
        cols = st.columns(min(4, len(siblings)))
        for idx, sib in enumerate(siblings):
            label = sib["label"]
            if sib.get("is_mainline"):
                label += " (main)"
            if sib.get("is_current"):
                label = "* " + label
            if cols[idx % len(cols)].button(label, key=f"{key_prefix}_sib_{sib['node_id']}", use_container_width=True):
                tree.set_current(sib["node_id"])
                changed = True

    next_vars = tree.variations_from_current()
    if len(next_vars) > 1:
        st.caption("Forward choices from this position:")
        cols = st.columns(min(4, len(next_vars)))
        for idx, var in enumerate(next_vars):
            label = var["label"] + (" (main)" if var["is_mainline"] else "")
            if cols[idx % len(cols)].button(label, key=f"{key_prefix}_child_{var['node_id']}", use_container_width=True):
                tree.step_forward(child_id=var["node_id"])
                changed = True
    return changed


def compare_nav_controls(main_tree: AnalysisTree, similar_tree: AnalysisTree) -> bool:
    changed = False
    st.markdown("**Compare Sync Controls**")
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Both Start", key="compare_start", use_container_width=True):
        main_tree.jump_to_start()
        similar_tree.jump_to_start()
        changed = True
    if c2.button("Both Back", key="compare_back", use_container_width=True):
        changed = main_tree.step_back() or similar_tree.step_back() or changed
    if c3.button("Both Forward", key="compare_forward", use_container_width=True):
        changed = main_tree.step_forward() or similar_tree.step_forward() or changed
    if c4.button("Both End", key="compare_end", use_container_width=True):
        main_tree.jump_to_end()
        similar_tree.jump_to_end()
        changed = True
    if c5.button("Match Similar Ply", key="compare_match_ply", use_container_width=True):
        similar_tree.set_current_to_ply_on_mainline(main_tree.current_ply_index)
        changed = True
    return changed


def render_compare_panel(*, orientation: str, similar_tree: AnalysisTree | None, size: int = 420) -> None:
    if not st.session_state.get("compare_mode_active") or similar_tree is None:
        return

    compare_sync_orientation = bool(st.session_state.get("compare_sync_orientation", True))
    compare_orientation = orientation
    if not compare_sync_orientation:
        override = str(st.session_state.get("compare_orientation_override", orientation)).strip().lower()
        compare_orientation = "black" if override == "black" else "white"

    st.markdown('<div class="compare-panel">', unsafe_allow_html=True)
    st.markdown("**Side-by-Side Position Compare**")
    if compare_nav_controls(st.session_state.main_tree, similar_tree):
        bump_main_board_instance()
        bump_similar_board_instance()
        sync_fen_input_to_main()
        st.rerun()

    compare_left = st.session_state.main_tree.current_board()
    compare_right = similar_tree.current_board()
    changed_squares = board_difference_squares(compare_left, compare_right)
    state_deltas = compare_state_notes(compare_left, compare_right)
    left_col, right_col = st.columns(2, gap="medium")
    with left_col:
        st.caption("Main board")
        render_large_compare_board(
            compare_left,
            orientation=compare_orientation,
            size=size,
            highlight_squares=changed_squares,
        )
        st.caption(f"Ply {st.session_state.main_tree.current_ply_index}")
    with right_col:
        st.caption("Selected similar board")
        render_large_compare_board(
            compare_right,
            orientation=compare_orientation,
            size=size,
            highlight_squares=changed_squares,
        )
        st.caption(f"Ply {similar_tree.current_ply_index}")

    st.markdown('<div class="compare-note">Changed squares are highlighted on both boards.</div>', unsafe_allow_html=True)
    if changed_squares:
        st.markdown(
            "".join(f'<span class="square-pill">{sq}</span>' for sq in changed_squares[:16]),
            unsafe_allow_html=True,
        )
    if state_deltas:
        render_chip_row(state_deltas)
    if not changed_squares and not state_deltas:
        st.caption("These two positions have the same visible board layout and state bits.")
    st.markdown("</div>", unsafe_allow_html=True)


def result_summary_line(meta: dict[str, Any]) -> str:
    headers = meta.get("headers", {}) or {}
    white = headers.get("White", "White")
    black = headers.get("Black", "Black")
    opening = headers.get("Opening", headers.get("ECO", ""))
    return f"{white} vs {black}" + (f" | {opening}" if opening else "")


def load_hit_into_views(
    hit_meta: dict[str, Any],
    bundle: CorpusBundle,
    *,
    load_into_main: bool,
    activate_compare: bool = False,
) -> None:
    game_id = hit_meta.get("game_id")
    if not game_id or game_id not in bundle.games_by_uid:
        set_notice("error", "Unable to locate full game for selected similar result.")
        return
    game = bundle.games_by_uid[game_id]
    ply_index = int(hit_meta.get("ply_index", 0))
    load_similar_tree_from_game(game, current_ply=ply_index)
    st.session_state.similar_selected_meta = dict(hit_meta)
    if activate_compare:
        st.session_state.compare_mode_active = True
    if load_into_main:
        push_main_snapshot(reason=f"Loaded similar result {game_id} ply {ply_index}")
        load_main_tree_from_game(game, current_ply=ply_index)
        set_notice("success", "Loaded similar result into main board and similar-game viewer.")
    else:
        set_notice("success", "Loaded similar result into similar-game viewer. Main board preserved.")


def compute_engine_eval_if_needed(board: chess.Board, *, stockfish_path: str, depth: int, need_eval: bool) -> EngineEval:
    if not need_eval:
        return EngineEval(ok=False, error=None)
    return analyze_position(board, stockfish_path=stockfish_path, depth=depth)


def encode_board_for_profile(
    board: chess.Board,
    *,
    encoder: PositionEncoder,
    stockfish_path: str,
    eval_depth: int,
    use_engine_eval: bool,
    eval_override: tuple[int | None, int | None] | None = None,
) -> tuple[np.ndarray, EngineEval]:
    eval_info = EngineEval(ok=False, error=None)
    cp = None
    mate = None
    if encoder.config.include_eval_score:
        if eval_override is not None:
            cp, mate = eval_override
            if cp is not None or mate is not None:
                norm_stm = normalize_eval_for_embedding(cp=cp, mate=mate)
                norm_white = norm_stm if board.turn == chess.WHITE else -norm_stm
                eval_info = EngineEval(
                    ok=True,
                    cp_stm=cp,
                    mate_stm=mate,
                    normalized_stm=norm_stm,
                    normalized_white=norm_white,
                    eval_bar_white=eval_bar_fraction_from_normalized_white(norm_white),
                    numeric_stm=format_eval_numeric(cp=cp, mate=mate),
                )
        elif use_engine_eval:
            eval_info = analyze_position(board, stockfish_path=stockfish_path, depth=eval_depth)
            if eval_info.ok:
                cp = eval_info.cp_stm
                mate = eval_info.mate_stm
    return encoder.encode(board, eval_cp=cp, eval_mate=mate), eval_info


def method_config_from_selection(
    method: str,
    *,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    ivf_nlist: int,
    ivf_nprobe: int,
    ivfpq_m: int,
    ivfpq_nbits: int,
) -> dict[str, Any]:
    if method == "faiss_hnsw":
        return {"M": int(hnsw_m), "ef_construction": int(hnsw_ef_construction), "ef_search": int(hnsw_ef_search)}
    if method == "faiss_ivf":
        return {"nlist": int(ivf_nlist), "nprobe": int(ivf_nprobe)}
    if method == "faiss_ivfpq":
        return {"nlist": int(ivf_nlist), "nprobe": int(ivf_nprobe), "pq_m": int(ivfpq_m), "pq_nbits": int(ivfpq_nbits)}
    return {}


def band_badge_html(band: str, score: float) -> str:
    palette = {
        "perfect": ("#065f46", "#d1fae5"),
        "high": ("#0f766e", "#ccfbf1"),
        "medium": ("#92400e", "#fef3c7"),
        "low": ("#9f1239", "#ffe4e6"),
    }
    fg, bg = palette.get(band, ("#1f2937", "#e5e7eb"))
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'background:{bg};color:{fg};font-weight:700;font-size:0.78rem;">'
        f"{band.upper()} ({score:.2f})</span>"
    )


def build_corpus_for_source(
    *,
    corpus_source_mode: str,
    pgn_folder_paths: list[str],
    encoder_cfg_json: str,
    every_n_plies: int,
    min_ply: int,
    max_positions: int,
    dedupe_games: bool,
    dedupe_fen: bool,
    use_engine_eval_for_corpus: bool,
    stockfish_path: str,
    eval_depth: int,
) -> CorpusBundle:
    if corpus_source_mode == "bundled_bulk":
        if not pgn_folder_paths:
            raise ValueError("No PGN files found in the bundled bulk dataset.")
        return build_corpus_from_paths_cached(
            json.dumps(sorted(pgn_folder_paths)),
            encoder_cfg_json,
            every_n_plies=every_n_plies,
            min_ply=min_ply,
            max_positions=max_positions,
            dedupe_games=dedupe_games,
            dedupe_fen=dedupe_fen,
            use_engine_eval_for_corpus=use_engine_eval_for_corpus,
            stockfish_path=stockfish_path,
            eval_depth=eval_depth,
        )
    return build_corpus_cached(
        str(SAMPLE_GAMES_DIR),
        encoder_cfg_json,
        every_n_plies=every_n_plies,
        min_ply=min_ply,
        max_positions=max_positions,
        dedupe_games=dedupe_games,
        dedupe_fen=dedupe_fen,
        use_engine_eval_for_corpus=use_engine_eval_for_corpus,
        stockfish_path=stockfish_path,
        eval_depth=eval_depth,
    )


def process_main_board_move_event(move_event: dict[str, Any] | None, *, analysis_mode: bool) -> bool:
    if not move_event:
        return False
    instance_id = int(move_event.get("instance_id", -1))
    if instance_id != st.session_state.main_board_instance_id:
        return False

    size_changed = False
    board_px = _as_optional_int(move_event.get("board_px"))
    if board_px is not None:
        board_px = max(220, min(2000, int(board_px)))
        prev_px = int(st.session_state.get("main_board_render_px", MAIN_BOARD_HEIGHT_PX))
        if board_px != prev_px:
            st.session_state.main_board_render_px = board_px
            size_changed = True

    uci = move_event.get("uci")
    if not uci:
        return size_changed
    event_signature = (
        instance_id,
        str(move_event.get("prev_fen") or ""),
        str(move_event.get("uci") or ""),
        str(move_event.get("fen") or ""),
    )
    if event_signature == st.session_state.main_last_event_signature:
        return False
    st.session_state.main_last_event_signature = event_signature

    tree: AnalysisTree = st.session_state.main_tree
    result = tree.play_move_uci(str(uci), allow_branching=analysis_mode, source="user")
    if not result.accepted:
        set_notice("warning", result.reason)
        bump_main_board_instance()
    elif result.created_branch:
        set_notice("info", "New branch variation created from the current analysis position.")
    return True


def main() -> None:
    ensure_session_state()
    inject_styles()
    st.markdown(
        """
        <div class="cap-hero">
          <h1>Similarity Search at Scale - Chess Position Demo</h1>
          <p>Benchmark exact and FAISS search methods on chess-position embeddings with interactive analysis, Stockfish eval, and branch-aware navigation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_notice_once()

    presets = load_presets_cached(str(PRESETS_PATH))
    sample_games = load_sample_games_cached(str(SAMPLE_GAMES_DIR))
    sample_game_ids = list(sample_games.keys())

    with st.sidebar:
        st.header("Load Positions")
        preset_names = [p["name"] for p in presets]
        preset_choice = st.selectbox("Curated presentation position (FEN)", preset_names, index=0 if preset_names else None)
        if st.button("Load Preset Position", use_container_width=True):
            preset = next((p for p in presets if p["name"] == preset_choice), None)
            if preset:
                st.session_state.main_tree = AnalysisTree(preset["fen"], headers={"Event": preset["name"], "Result": "*"})
                st.session_state.main_game_id = None
                clear_main_restore_stack()
                bump_main_board_instance()
                sync_fen_input_to_main()
                set_notice("success", f"Loaded preset position: {preset['name']}")
                st.rerun()
        if preset_choice:
            preset = next((p for p in presets if p["name"] == preset_choice), None)
            if preset and preset.get("description"):
                st.caption(preset["description"])

        game_choice = st.selectbox(
            "Curated sample game (PGN)",
            sample_game_ids,
            format_func=lambda gid: sample_games[gid].title(),
            index=0 if sample_game_ids else None,
        )
        if st.button("Load Sample Game", use_container_width=True):
            if game_choice:
                load_main_tree_from_game(sample_games[game_choice], current_ply=0, clear_restore_stack_flag=True)
                set_notice("success", "Loaded sample game into main board.")
                st.rerun()

        st.divider()
        st.subheader("Milestone 1 Controls")
        if st.session_state.pending_fen_input_text is not None:
            st.session_state.fen_input_text = st.session_state.pending_fen_input_text
            st.session_state.pending_fen_input_text = None
        fen_input = st.text_input("FEN input", value=st.session_state.main_tree.current_fen, key="fen_input_text")
        fen_ok, fen_err = validate_fen(fen_input)
        if not fen_ok and fen_input.strip():
            st.caption(f"Invalid FEN: {fen_err}")
        col_fen_a, col_fen_b = st.columns(2)
        if col_fen_a.button("Load FEN", use_container_width=True, disabled=not fen_ok):
            st.session_state.main_tree = AnalysisTree(fen_input, headers={"Event": "FEN Input", "Result": "*"})
            st.session_state.main_game_id = None
            clear_main_restore_stack()
            bump_main_board_instance()
            sync_fen_input_to_main()
            set_notice("success", "Loaded FEN into main board.")
            st.rerun()
        if col_fen_b.button("Reset Start", use_container_width=True):
            st.session_state.main_tree = AnalysisTree()
            st.session_state.main_game_id = None
            clear_main_restore_stack()
            bump_main_board_instance()
            sync_fen_input_to_main()
            set_notice("success", "Reset main board to starting position.")
            st.rerun()

        st.divider()
        st.header("Board & Engine")
        orientation = st.selectbox("Board orientation", ["white", "black"], index=0)
        analysis_mode = st.toggle("Analysis mode (branching enabled)", value=True)
        show_eval_bar = st.toggle("Show eval bar", value=True)
        show_best_move_arrow = st.toggle("Show best-move arrow", value=True)
        show_pv_line = st.toggle("Show PV line", value=True)
        stockfish_path = st.text_input("Stockfish path", value=resolved_default_stockfish_path())
        engine_depth = st.slider(
            "Stockfish depth",
            min_value=6,
            max_value=PUBLIC_MAX_ENGINE_DEPTH if PUBLIC_APP_MODE else 20,
            value=PUBLIC_DEFAULT_ENGINE_DEPTH if PUBLIC_APP_MODE else 10,
        )
        if not faiss_available():
            st.caption("FAISS not detected: brute-force search works; FAISS methods will return clear errors.")

        st.divider()
        st.header("Similarity Search")
        method_options = method_options_for_ui()
        search_method = st.selectbox("Search method", method_options, index=0)
        search_metric = st.selectbox("Metric", ["cosine", "l2"], index=0)
        top_k = st.slider("Top-k similar results", min_value=1, max_value=20, value=6)
        click_load_mode = st.radio(
            "Click behavior for similar results",
            options=["viewer_only", "main_and_viewer"],
            format_func=lambda v: "Load in similar viewer only" if v == "viewer_only" else "Load in main + viewer",
            index=0,
        )
        max_hits_per_game = st.slider("Max hits per game", min_value=1, max_value=5, value=2)
        min_ply_gap_same_game = st.slider("Min ply gap for same-game hits", min_value=0, max_value=30, value=3)
        exclude_same_game = st.checkbox("Exclude hits from the same loaded game", value=False)
        result_min_ply = st.slider("Minimum ply for returned hits", min_value=0, max_value=40, value=4)
        phase_filter = st.selectbox("Game phase filter", ["all", "opening", "middlegame", "endgame"], index=0)
        diversify_positions = st.checkbox("Diversify near-duplicate board layouts", value=True)
        st.caption("Corpus source")
        corpus_source_mode = st.radio(
            "Corpus source",
            options=["curated_samples", "bundled_bulk"],
            format_func=lambda v: "Curated sample PGNs" if v == "curated_samples" else "Bundled bulk PGNs",
            index=0,
        )
        pgn_folder_paths: list[str] = []
        if corpus_source_mode == "bundled_bulk":
            pgn_folder_paths = list_pgn_paths_cached(str(DEFAULT_BULK_PGN_DIR), True)
            st.caption(f"Detected PGN files: {len(pgn_folder_paths)}")
        st.caption("Embedding features (deterministic per position)")
        enc_include_eval = st.checkbox("Include engine eval feature", value=False)
        enc_include_piece_counts = st.checkbox("Include piece count features", value=True)
        enc_include_piece_info = st.checkbox("Include piece info features", value=True)
        enc_include_material = st.checkbox("Include material balance features", value=True)
        corpus_every_n = st.slider("Corpus sampling (every N plies)", min_value=1, max_value=6, value=2)
        corpus_min_ply = st.slider("Minimum ply in corpus", min_value=0, max_value=20, value=1)
        corpus_max_positions = st.number_input(
            "Max corpus positions (demo)",
            min_value=20,
            max_value=PUBLIC_MAX_CORPUS_POSITIONS if PUBLIC_APP_MODE else 300000,
            value=PUBLIC_DEFAULT_CORPUS_MAX_POSITIONS if PUBLIC_APP_MODE else 800,
            step=20,
        )
        corpus_dedupe_games = st.checkbox("Dedupe duplicate games across PGNs", value=True)
        corpus_dedupe_fen = st.checkbox("Dedupe identical FENs in corpus", value=False)
        corpus_use_engine_eval = st.checkbox("Use Stockfish eval while building corpus vectors", value=False)
        exclude_exact_fen = st.checkbox("Exclude exact-FEN matches from results", value=True)
        with st.expander("FAISS HNSW / IVF / IVFPQ knobs", expanded=False):
            hnsw_m = st.number_input("HNSW M", min_value=4, max_value=64, value=32, step=4)
            hnsw_ef_construction = st.number_input("HNSW ef_construction", min_value=20, max_value=500, value=200, step=10)
            hnsw_ef_search = st.number_input("HNSW ef_search", min_value=8, max_value=256, value=64, step=8)
            ivf_nlist = st.number_input("IVF nlist", min_value=1, max_value=1024, value=32, step=1)
            ivf_nprobe = st.number_input("IVF nprobe", min_value=1, max_value=256, value=8, step=1)
            ivfpq_m = st.number_input("IVFPQ pq_m", min_value=1, max_value=64, value=8, step=1)
            ivfpq_nbits = st.number_input("IVFPQ pq_nbits", min_value=4, max_value=12, value=8, step=1)
        query_auto_refresh = st.checkbox("Auto-refresh similar results as board changes", value=True)

    encoder_cfg = EncoderConfig(
        include_eval_score=enc_include_eval,
        include_piece_counts=enc_include_piece_counts,
        include_piece_info=enc_include_piece_info,
        include_material_balance=enc_include_material,
    )
    encoder = PositionEncoder(encoder_cfg)

    engine_needed_for_ui = show_eval_bar or show_best_move_arrow or show_pv_line
    engine_needed_for_query_embedding = encoder_cfg.include_eval_score
    need_engine_eval_now = engine_needed_for_ui or engine_needed_for_query_embedding
    main_tree: AnalysisTree = st.session_state.main_tree

    tabs = st.tabs(["Analysis Demo", "Demo Lab", "Benchmark"])

    with tabs[0]:
        main_col, side_col = st.columns([1.2, 0.85], gap="large")

        with main_col:
            if tree_nav_controls(main_tree, key_prefix="main_nav", title="Main Board Navigation"):
                st.rerun()

            main_board = main_tree.current_board()
            eval_info = compute_engine_eval_if_needed(
                main_board,
                stockfish_path=stockfish_path,
                depth=engine_depth,
                need_eval=need_engine_eval_now,
            )
            if need_engine_eval_now and not eval_info.ok and stockfish_path.strip():
                st.caption(f"Engine unavailable: {eval_info.error}")

            best_move_arrows: list[dict[str, str]] = []
            if show_best_move_arrow and eval_info.ok and eval_info.best_move_uci:
                best_move_arrows = [
                    {
                        "orig": eval_info.best_move_uci[:2],
                        "dest": eval_info.best_move_uci[2:4],
                        "brush": "green",
                    }
                ]

            if show_eval_bar:
                eval_bar_height_px = int(st.session_state.get("main_board_render_px", MAIN_BOARD_HEIGHT_PX))
                eval_bar_height_px = max(220, min(2000, eval_bar_height_px))
                bar_col, board_col = st.columns([0.12, 0.88], gap="small")
                with bar_col:
                    render_eval_bar(eval_info, height_px=eval_bar_height_px)
                with board_col:
                    move_event = cg_board(
                        main_board,
                        key="main_cg_board",
                        orientation=orientation,
                        height=MAIN_BOARD_HEIGHT_PX,
                        disabled=False,
                        last_move_uci=main_tree.current.move_uci,
                        arrows=best_move_arrows,
                        instance_id=st.session_state.main_board_instance_id,
                    )
            else:
                move_event = cg_board(
                    main_board,
                    key="main_cg_board",
                    orientation=orientation,
                    height=MAIN_BOARD_HEIGHT_PX,
                    disabled=False,
                    last_move_uci=main_tree.current.move_uci,
                    arrows=best_move_arrows,
                    instance_id=st.session_state.main_board_instance_id,
                )

            if process_main_board_move_event(move_event, analysis_mode=analysis_mode):
                st.rerun()

            st.markdown('<div class="cap-card">', unsafe_allow_html=True)
            st.markdown("**Branching / Variation Context**")
            if analysis_mode:
                st.markdown('<span class="branch-pill">Analysis mode: branching enabled</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="branch-pill">Strict game mode: branching disabled</span>', unsafe_allow_html=True)
            st.caption("If you step backward and play a new move in analysis mode, a branch is created and the original line is preserved.")
            line_preview = " ".join(main_tree.current_line_san()[-12:]) or "(start position)"
            st.code(line_preview, language=None)
            st.markdown("</div>", unsafe_allow_html=True)

            render_compare_panel(
                orientation=orientation,
                similar_tree=st.session_state.get("similar_tree"),
                size=420,
            )

            with st.expander("Export Current Analysis Tree as PGN (with variations)", expanded=False):
                st.text_area("PGN", value=main_tree.to_pgn_string(), height=220, key="pgn_export_area")

        with side_col:
            st.markdown('<div class="cap-card">', unsafe_allow_html=True)
            st.markdown("**Engine Evaluation**")
            if eval_info.ok:
                c1, c2 = st.columns(2)
                c1.markdown('<div class="cap-label">Eval (STM)</div>', unsafe_allow_html=True)
                c1.markdown(f'<div class="cap-value">{eval_info.numeric_stm}</div>', unsafe_allow_html=True)
                c2.markdown('<div class="cap-label">Depth</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="cap-value">{eval_info.depth or "-"}</div>', unsafe_allow_html=True)
                st.caption(f"Best move: `{eval_info.best_move_uci or '-'}`")
                if show_pv_line and eval_info.pv_san:
                    st.caption("PV line")
                    st.code(" ".join(eval_info.pv_san), language=None)
            else:
                st.caption("Engine eval disabled or unavailable.")
                if eval_info.error:
                    st.caption(eval_info.error)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="cap-card">', unsafe_allow_html=True)
            st.markdown("**Current Position Summary**")
            st.caption(f"FEN: `{main_tree.current_fen}`")
            st.caption(f"Ply index: {main_tree.current_ply_index}")
            st.caption(f"Variations from current node: {len(main_tree.current.children)}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="cap-card">', unsafe_allow_html=True)
            st.markdown("**Similarity Search Results**")
            if encoder_cfg.include_eval_score and not corpus_use_engine_eval:
                st.caption("Eval feature is enabled, but corpus vectors use zero eval unless corpus engine-eval preprocessing is enabled.")

            encoder_cfg_json = json.dumps(encoder_cfg.to_dict(), sort_keys=True)
            corpus_bundle: CorpusBundle | None = None
            try:
                source_label = "bundled bulk PGNs" if corpus_source_mode == "bundled_bulk" else "curated sample PGNs"
                with st.spinner(f"Preparing in-memory corpus from {source_label}..."):
                    corpus_bundle = build_corpus_for_source(
                        corpus_source_mode=corpus_source_mode,
                        pgn_folder_paths=pgn_folder_paths,
                        encoder_cfg_json=encoder_cfg_json,
                        every_n_plies=corpus_every_n,
                        min_ply=corpus_min_ply,
                        max_positions=int(corpus_max_positions),
                        dedupe_games=corpus_dedupe_games,
                        dedupe_fen=corpus_dedupe_fen,
                        use_engine_eval_for_corpus=bool(corpus_use_engine_eval),
                        stockfish_path=stockfish_path,
                        eval_depth=max(6, engine_depth - 2),
                    )
            except Exception as exc:
                st.error(f"Corpus build failed: {type(exc).__name__}: {exc}")

            method_cfg: dict[str, Any] = {}
            if search_method == "faiss_hnsw":
                method_cfg = {
                    "M": int(hnsw_m),
                    "ef_construction": int(hnsw_ef_construction),
                    "ef_search": int(hnsw_ef_search),
                }
            elif search_method == "faiss_ivf":
                method_cfg = {"nlist": int(ivf_nlist), "nprobe": int(ivf_nprobe)}
            elif search_method == "faiss_ivfpq":
                method_cfg = {
                    "nlist": int(ivf_nlist),
                    "nprobe": int(ivf_nprobe),
                    "pq_m": int(ivfpq_m),
                    "pq_nbits": int(ivfpq_nbits),
                }

            if corpus_bundle is not None:
                st.caption(
                    f"Corpus: {len(corpus_bundle.positions)} positions - {len(corpus_bundle.games_by_uid)} games - dim={corpus_bundle.vectors.shape[1]}"
                )
                skipped_dup_games = int(corpus_bundle.manifest.get("duplicate_games_skipped", 0))
                if skipped_dup_games > 0:
                    st.caption(f"Skipped duplicate games: {skipped_dup_games}")
                try:
                    index_cfg_json = json.dumps(method_cfg, sort_keys=True)
                    search_index = build_search_index_cached(search_method, search_metric, corpus_bundle.vectors, index_cfg_json)
                    mem_kb = (search_index.memory_bytes_est or 0) / 1024.0 if search_index.memory_bytes_est else None
                    st.caption(
                        f"Index `{search_index.method}` build time: {search_index.build_time_ms:.1f} ms"
                        + (f" - estimated index footprint ~{mem_kb:.1f} KB" if mem_kb is not None else "")
                    )

                    should_query = query_auto_refresh or st.button("Refresh Similar Results", use_container_width=True)
                    if should_query and len(corpus_bundle.positions) > 0:
                        q_eval_cp = eval_info.cp_stm if (encoder_cfg.include_eval_score and eval_info.ok) else None
                        q_eval_mate = eval_info.mate_stm if (encoder_cfg.include_eval_score and eval_info.ok) else None
                        query_vec = encoder.encode(main_board, eval_cp=q_eval_cp, eval_mate=q_eval_mate)
                        query_snapshot = board_feature_snapshot(
                            main_board,
                            eval_cp_stm=q_eval_cp,
                            eval_mate_stm=q_eval_mate,
                        )

                        exclude_indices: set[int] = set()
                        if exclude_exact_fen:
                            current_fen = main_board.fen(en_passant="fen")
                            for pos in corpus_bundle.positions:
                                if pos.get("fen") == current_fen:
                                    exclude_indices.add(int(pos["corpus_index"]))
                        excluded_game_ids: set[str] = set()
                        if exclude_same_game and st.session_state.get("main_game_id"):
                            excluded_game_ids.add(str(st.session_state.main_game_id))

                        hits = hits_with_metadata(
                            search_index,
                            query_vec,
                            corpus_bundle.positions,
                            k=top_k,
                            exclude_indices=exclude_indices if exclude_indices else None,
                            max_hits_per_game=max_hits_per_game,
                            min_ply_gap_same_game=min_ply_gap_same_game,
                            exclude_game_ids=excluded_game_ids if excluded_game_ids else None,
                            min_result_ply=result_min_ply,
                            phase_filter=phase_filter,
                            diversify_positions=diversify_positions,
                        )

                        if not hits:
                            st.caption("No similar results found.")
                        for hit in hits:
                            meta = hit.metadata
                            hit_board = chess.Board(str(meta.get("fen", "")))
                            hit_snapshot = board_feature_snapshot(
                                hit_board,
                                eval_cp_stm=_as_optional_int(meta.get("eval_cp_stm")),
                                eval_mate_stm=_as_optional_int(meta.get("eval_mate_stm")),
                            )
                            hit_vec = np.asarray(corpus_bundle.vectors[int(hit.index)], dtype=np.float32)
                            hit_explain = explain_pair(query_vec, hit_vec, group_slices=encoder.feature_slices())
                            hit_score = float(hit_explain["score_100"])
                            hit_band = score_band(hit_score, default_score_bands())
                            hit_chips = explanation_chips(hit_explain, query_snapshot, hit_snapshot, limit=4)
                            hit_reason = build_similarity_english_summary(
                                score=hit_score,
                                band=hit_band,
                                pair_explain=hit_explain,
                                query_snapshot=query_snapshot,
                                candidate_snapshot=hit_snapshot,
                            )
                            label = f"#{hit.rank} - d={hit.distance:.4f} - ply {meta.get('ply_index', '?')} - {result_summary_line(meta)}"
                            st.markdown('<div class="similar-result-row">', unsafe_allow_html=True)
                            thumb_col, txt_col = st.columns([0.5, 0.5], gap="small")
                            with thumb_col:
                                render_mini_board_thumbnail(str(meta.get("fen", "")), orientation=orientation, size=188)
                            with txt_col:
                                st.markdown(f"**{label}**", unsafe_allow_html=False)
                                st.markdown(band_badge_html(hit_band, hit_score), unsafe_allow_html=True)
                                render_chip_row(hit_chips)
                                st.caption(hit_reason)
                                st.caption(
                                    f"{meta.get('source_pgn_file', '')} | result={meta.get('result', '*')} | "
                                    f"game={meta.get('game_id', '')} | phase={meta.get('phase', 'n/a')}"
                                )
                                btn_a, btn_b = st.columns(2)
                                if btn_a.button("Load", key=f"hit_btn_{hit.index}", use_container_width=True):
                                    load_hit_into_views(
                                        meta,
                                        corpus_bundle,
                                        load_into_main=(click_load_mode == "main_and_viewer"),
                                        activate_compare=False,
                                    )
                                    st.rerun()
                                if btn_b.button("Compare with current board", key=f"hit_cmp_{hit.index}", use_container_width=True):
                                    load_hit_into_views(
                                        meta,
                                        corpus_bundle,
                                        load_into_main=False,
                                        activate_compare=True,
                                    )
                                    st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"Search error: {type(exc).__name__}: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="cap-card">', unsafe_allow_html=True)
            st.markdown("**Loaded Similar Game Viewer**")
            similar_tree: AnalysisTree | None = st.session_state.similar_tree
            if similar_tree is None:
                st.caption("Click a similar result above to load the full game and jump to the matched move.")
            else:
                if tree_nav_controls(similar_tree, key_prefix="similar_nav", title="Similar Game Navigation"):
                    st.rerun()
                cg_board(
                    similar_tree.current_board(),
                    key="similar_cg_board",
                    orientation=orientation,
                    height=360,
                    disabled=True,
                    last_move_uci=similar_tree.current.move_uci,
                    arrows=[],
                    instance_id=st.session_state.similar_board_instance_id,
                )
                meta = st.session_state.get("similar_selected_meta") or {}
                if meta:
                    st.caption(
                        f"Selected hit: {meta.get('source_pgn_file', '')} - game={meta.get('game_id', '')} - matched ply={meta.get('ply_index', '')}"
                    )
                if st.button("Load Current Similar View Into Main Board", use_container_width=True):
                    push_main_snapshot(reason="Synced similar viewer into main board")
                    sync_similar_into_main()
                    set_notice("success", "Copied the similar viewer position into the main board.")
                    st.rerun()
                compare_enabled = st.toggle(
                    "Side-by-side compare mode",
                    value=bool(st.session_state.get("compare_mode_active", False)),
                    key="compare_mode_active",
                )
                if compare_enabled:
                    st.checkbox(
                        "Sync orientation with the main board",
                        value=True,
                        key="compare_sync_orientation",
                    )
                    if not st.session_state.get("compare_sync_orientation", True):
                        st.selectbox(
                            "Compare panel orientation",
                            ["white", "black"],
                            index=0 if orientation == "white" else 1,
                            key="compare_orientation_override",
                        )
                    st.caption("The full compare boards render in the main left column for a wider side-by-side view.")
            if st.session_state.main_restore_stack:
                last_label = st.session_state.main_restore_stack[-1].get("label", "previous context")
                st.caption(f"Previous main context available: {last_label}")
                if st.button("Return To Previous Main Game", use_container_width=True):
                    if restore_previous_main_snapshot():
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("Demo Lab: Score Bands + Vector Explainability")
        st.caption(
            "Uses bundled demo positions and a custom demo PGN to present perfect/high/low scores and feature-level similarity reasons."
        )

        demo_manifest = load_demo_scenarios_cached(str(DEMO_SCENARIOS_PATH))
        scenario_rows = demo_manifest.get("scenarios", [])
        if not scenario_rows:
            st.error("No demo scenarios found. Add `data/demo/demo_scenarios.json` with at least one scenario.")
        else:
            profile_map = demo_profile_configs()
            profile_names = list(profile_map.keys())
            demo_method_options = method_options_for_ui()

            ctrl_a, ctrl_b = st.columns([1.05, 0.95], gap="large")
            with ctrl_a:
                scenario_ids = [str(s.get("id", f"s{i+1}")) for i, s in enumerate(scenario_rows)]
                scenario_lookup = {str(s.get("id", f"s{i+1}")): s for i, s in enumerate(scenario_rows)}
                selected_scenario_id = st.selectbox(
                    "Demo scenario",
                    scenario_ids,
                    format_func=lambda sid: f"{sid} - {scenario_lookup[sid].get('title', sid)}",
                )
                selected_scenario = scenario_lookup[selected_scenario_id]
                default_method = str(demo_manifest.get("default_method", "brute"))
                method_index = demo_method_options.index(default_method) if default_method in demo_method_options else 0
                demo_method = st.selectbox("Demo retrieval method", demo_method_options, index=method_index, key="demo_method")
                demo_top_k = st.slider(
                    "Demo top-k",
                    min_value=1,
                    max_value=20,
                    value=int(demo_manifest.get("default_top_k", 6)),
                    key="demo_top_k",
                )
                demo_profile_name = st.selectbox("Retrieval profile", profile_names, index=2, key="demo_profile_name")
                demo_exclude_exact = st.checkbox("Exclude exact query FEN from demo results", value=False, key="demo_exclude_exact")
                demo_use_engine_eval = st.checkbox(
                    "Use Stockfish when profile includes eval feature",
                    value=True,
                    key="demo_use_engine_eval",
                )

            with ctrl_b:
                st.markdown("**Score band editor (0-100)**")
                default_bands = normalize_score_bands(demo_manifest.get("default_score_bands"))
                perfect_min = st.number_input("Perfect minimum", min_value=0.0, max_value=100.0, value=float(default_bands["perfect"][0]), step=0.5)
                high_min = st.number_input("High minimum", min_value=0.0, max_value=100.0, value=float(default_bands["high"][0]), step=0.5)
                medium_min = st.number_input("Medium minimum", min_value=0.0, max_value=100.0, value=float(default_bands["medium"][0]), step=0.5)
                perfect_floor = max(0.0, min(100.0, float(perfect_min)))
                high_floor = max(0.0, min(perfect_floor, float(high_min)))
                medium_floor = max(0.0, min(high_floor, float(medium_min)))
                active_bands = {
                    "perfect": (perfect_floor, 100.0),
                    "high": (high_floor, max(high_floor, perfect_floor - 0.01)),
                    "medium": (medium_floor, max(medium_floor, high_floor - 0.01)),
                    "low": (0.0, max(0.0, medium_floor - 0.01)),
                }
                st.caption(
                    f"Active bands: perfect {active_bands['perfect'][0]:.2f}-100 | "
                    f"high {active_bands['high'][0]:.2f}-{active_bands['high'][1]:.2f} | "
                    f"medium {active_bands['medium'][0]:.2f}-{active_bands['medium'][1]:.2f} | "
                    f"low {active_bands['low'][0]:.2f}-{active_bands['low'][1]:.2f}"
                )

            retrieval_cfg = profile_map[demo_profile_name]
            retrieval_encoder = PositionEncoder(retrieval_cfg)
            demo_bundle = build_demo_dataset_cached(
                str(DEMO_GAMES_PATH),
                str(DEMO_POSITIONS_PATH),
                json.dumps(retrieval_cfg.to_dict(), sort_keys=True),
                stockfish_path=stockfish_path,
                eval_depth=max(8, engine_depth),
                use_engine_eval_for_eval_feature=bool(demo_use_engine_eval),
            )
            st.caption(
                f"Demo dataset: {len(demo_bundle['positions'])} positions from "
                f"{len(demo_bundle['games_by_uid'])} entries (positions + demo PGN)."
            )

            try:
                query_fen, query_label, query_ref_meta = resolve_demo_ref(
                    selected_scenario.get("query", {}),
                    positions_by_id=demo_bundle["positions_by_id"],
                    games_by_uid=demo_bundle["games_by_uid"],
                )
                candidate_fen, candidate_label, candidate_ref_meta = resolve_demo_ref(
                    selected_scenario.get("candidate", {}),
                    positions_by_id=demo_bundle["positions_by_id"],
                    games_by_uid=demo_bundle["games_by_uid"],
                )
            except Exception as exc:
                st.error(f"Scenario reference error: {type(exc).__name__}: {exc}")
                query_fen = None
                candidate_fen = None

            if query_fen and candidate_fen:
                query_board = chess.Board(query_fen)
                candidate_board = chess.Board(candidate_fen)
                query_eval_override = eval_override_from_ref_meta(query_ref_meta)
                candidate_eval_override = eval_override_from_ref_meta(candidate_ref_meta)

                action_col_a, action_col_b = st.columns([0.45, 0.55], gap="medium")
                if action_col_a.button("Load Scenario Query Into Main Board", use_container_width=True):
                    st.session_state.main_tree = AnalysisTree(query_board.fen(en_passant="fen"), headers={"Event": f"Demo {selected_scenario_id}", "Result": "*"})
                    st.session_state.main_game_id = None
                    clear_main_restore_stack()
                    bump_main_board_instance()
                    sync_fen_input_to_main()
                    set_notice("success", f"Loaded demo scenario {selected_scenario_id} query into main board.")
                    st.rerun()
                action_col_b.caption(
                    f"Scenario focus: {', '.join(selected_scenario.get('feature_focus', [])) or 'n/a'} | "
                    f"Expected band: {selected_scenario.get('expected_band', 'n/a')}"
                )
                if query_eval_override is not None or candidate_eval_override is not None:
                    st.caption("Using curated demo eval annotations for deterministic eval-feature behavior.")

                method_cfg = method_config_from_selection(
                    demo_method,
                    hnsw_m=int(hnsw_m),
                    hnsw_ef_construction=int(hnsw_ef_construction),
                    hnsw_ef_search=int(hnsw_ef_search),
                    ivf_nlist=int(ivf_nlist),
                    ivf_nprobe=int(ivf_nprobe),
                    ivfpq_m=int(ivfpq_m),
                    ivfpq_nbits=int(ivfpq_nbits),
                )
                demo_index = build_search_index_cached(
                    demo_method,
                    "cosine",
                    demo_bundle["vectors"],
                    json.dumps(method_cfg, sort_keys=True),
                )
                query_vec, query_eval = encode_board_for_profile(
                    query_board,
                    encoder=retrieval_encoder,
                    stockfish_path=stockfish_path,
                    eval_depth=max(8, engine_depth),
                    use_engine_eval=bool(demo_use_engine_eval),
                    eval_override=query_eval_override,
                )
                query_snapshot = board_feature_snapshot(
                    query_board,
                    eval_cp_stm=query_eval.cp_stm if query_eval.ok else None,
                    eval_mate_stm=query_eval.mate_stm if query_eval.ok else None,
                )
                exclude_indices: set[int] = set()
                if demo_exclude_exact:
                    q_fen_exact = query_board.fen(en_passant="fen")
                    for meta in demo_bundle["positions"]:
                        if str(meta.get("fen", "")) == q_fen_exact:
                            exclude_indices.add(int(meta.get("corpus_index", -1)))

                demo_hits = hits_with_metadata(
                    demo_index,
                    query_vec,
                    demo_bundle["positions"],
                    k=int(demo_top_k),
                    exclude_indices=exclude_indices if exclude_indices else None,
                )

                st.markdown("**Scenario Query vs Corpus Results**")
                st.caption(f"Query source: {query_label}")
                if not demo_hits:
                    st.warning("No demo hits found for this query.")
                for hit in demo_hits:
                    score = score_from_distance(hit.distance)
                    band = score_band(score, active_bands)
                    meta = hit.metadata
                    row_col_a, row_col_b = st.columns([0.44, 0.56], gap="small")
                    with row_col_a:
                        render_mini_board_thumbnail(str(meta.get("fen", "")), orientation=orientation, size=172)
                    with row_col_b:
                        st.markdown(
                            f"**#{hit.rank}** {band_badge_html(band, score)}",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"distance={hit.distance:.5f} | game={meta.get('game_id', '')} | "
                            f"ply={meta.get('ply_index', '')} | source={meta.get('source_pgn_file', '')}"
                        )
                        try:
                            hit_vec = np.asarray(demo_bundle["vectors"][int(hit.index)], dtype=np.float32)
                            hit_explain = explain_pair(
                                query_vec,
                                hit_vec,
                                group_slices=retrieval_encoder.feature_slices(),
                            )
                            hit_board = chess.Board(str(meta.get("fen", "")))
                            hit_snapshot = board_feature_snapshot(
                                hit_board,
                                eval_cp_stm=_as_optional_int(meta.get("eval_cp_stm")),
                                eval_mate_stm=_as_optional_int(meta.get("eval_mate_stm")),
                            )
                            render_chip_row(explanation_chips(hit_explain, query_snapshot, hit_snapshot, limit=4))
                            hit_reason = build_similarity_english_summary(
                                score=float(hit_explain["score_100"]),
                                band=band,
                                pair_explain=hit_explain,
                                query_snapshot=query_snapshot,
                                candidate_snapshot=hit_snapshot,
                            )
                            st.caption(hit_reason)
                        except Exception:
                            pass
                    st.markdown("---")

                st.markdown("**Why This Scenario Candidate Is Similar (or Not)**")
                candidate_vec, candidate_eval = encode_board_for_profile(
                    candidate_board,
                    encoder=retrieval_encoder,
                    stockfish_path=stockfish_path,
                    eval_depth=max(8, engine_depth),
                    use_engine_eval=bool(demo_use_engine_eval),
                    eval_override=candidate_eval_override,
                )
                pair_explain = explain_pair(
                    query_vec,
                    candidate_vec,
                    group_slices=retrieval_encoder.feature_slices(),
                )
                candidate_score = float(pair_explain["score_100"])
                candidate_band = score_band(candidate_score, active_bands)
                expected_band = str(selected_scenario.get("expected_band", "")).strip().lower()

                exp_col_a, exp_col_b = st.columns([0.5, 0.5], gap="medium")
                with exp_col_a:
                    st.caption(f"Candidate source: {candidate_label}")
                    st.markdown(band_badge_html(candidate_band, candidate_score), unsafe_allow_html=True)
                    st.caption(
                        f"raw distance={pair_explain['distance']:.6f} | similarity={pair_explain['similarity']:.6f}"
                    )
                    if expected_band:
                        if candidate_band == expected_band:
                            st.success(f"Expected band matched: {expected_band.upper()}")
                        else:
                            st.warning(f"Expected {expected_band.upper()}, observed {candidate_band.upper()}")
                    st.caption(selected_scenario.get("narrative", ""))
                    render_chip_row(explanation_chips(pair_explain, query_snapshot, board_feature_snapshot(
                        candidate_board,
                        eval_cp_stm=candidate_eval.cp_stm if candidate_eval.ok else None,
                        eval_mate_stm=candidate_eval.mate_stm if candidate_eval.ok else None,
                    ), limit=4))
                    render_mini_board_thumbnail(query_board.fen(en_passant="fen"), orientation=orientation, size=130)
                    render_mini_board_thumbnail(candidate_board.fen(en_passant="fen"), orientation=orientation, size=130)
                with exp_col_b:
                    contrib_df = pd.DataFrame(pair_explain["group_contributions"])
                    if not contrib_df.empty:
                        st.bar_chart(contrib_df.set_index("group")[["contribution"]], use_container_width=True)
                        st.dataframe(
                            contrib_df[["group", "contribution", "share_abs_pct", "share_signed_pct"]],
                            use_container_width=True,
                        )
                        top_group = contrib_df.iloc[contrib_df["share_abs_pct"].idxmax()]
                        st.caption(
                            f"Largest driver: {top_group['group']} "
                            f"(abs share {float(top_group['share_abs_pct']):.2f}%)."
                        )

                candidate_snapshot = board_feature_snapshot(
                    candidate_board,
                    eval_cp_stm=candidate_eval.cp_stm if candidate_eval.ok else None,
                    eval_mate_stm=candidate_eval.mate_stm if candidate_eval.ok else None,
                )
                candidate_reason = build_similarity_english_summary(
                    score=candidate_score,
                    band=candidate_band,
                    pair_explain=pair_explain,
                    query_snapshot=query_snapshot,
                    candidate_snapshot=candidate_snapshot,
                )
                st.info(candidate_reason)
                delta_df = pd.DataFrame(snapshot_delta_rows(query_snapshot, candidate_snapshot))
                st.markdown("**Interpretable Feature Delta Table**")
                st.dataframe(delta_df, use_container_width=True)

                st.markdown("**A/B/C Profile Score Comparison (Same Query/Candidate)**")
                comparison_rows: list[dict[str, Any]] = []
                for profile_name, profile_cfg in profile_map.items():
                    profile_encoder = PositionEncoder(profile_cfg)
                    q_vec_p, _ = encode_board_for_profile(
                        query_board,
                        encoder=profile_encoder,
                        stockfish_path=stockfish_path,
                        eval_depth=max(8, engine_depth),
                        use_engine_eval=bool(demo_use_engine_eval),
                        eval_override=query_eval_override,
                    )
                    c_vec_p, _ = encode_board_for_profile(
                        candidate_board,
                        encoder=profile_encoder,
                        stockfish_path=stockfish_path,
                        eval_depth=max(8, engine_depth),
                        use_engine_eval=bool(demo_use_engine_eval),
                        eval_override=candidate_eval_override,
                    )
                    profile_explain = explain_pair(q_vec_p, c_vec_p, group_slices=profile_encoder.feature_slices())
                    sc = float(profile_explain["score_100"])
                    comparison_rows.append(
                        {
                            "profile": profile_name,
                            "score_100": sc,
                            "distance": float(profile_explain["distance"]),
                            "band": score_band(sc, active_bands),
                        }
                    )
                comparison_df = pd.DataFrame(comparison_rows)
                st.dataframe(comparison_df, use_container_width=True)
                if {"A", "B"}.issubset(set(comparison_df["profile"])):
                    score_a = float(comparison_df.loc[comparison_df["profile"] == "A", "score_100"].iloc[0])
                    score_b = float(comparison_df.loc[comparison_df["profile"] == "B", "score_100"].iloc[0])
                    st.caption(
                        f"Profile B vs A: {score_b - score_a:+.2f} score points. "
                        "B adds piece-count, piece-info, and material features on top of the base 781 structural encoding."
                    )

                extra_refs = selected_scenario.get("comparison_candidates", [])
                if isinstance(extra_refs, list) and len(extra_refs) > 1:
                    st.markdown("**Ranking Shift Demonstrator (Profile-dependent ordering)**")
                    rank_rows: list[dict[str, Any]] = []
                    resolved_candidates: list[dict[str, Any]] = []
                    for ref in extra_refs:
                        if not isinstance(ref, dict):
                            continue
                        try:
                            fen_x, label_x, meta_x = resolve_demo_ref(
                                ref,
                                positions_by_id=demo_bundle["positions_by_id"],
                                games_by_uid=demo_bundle["games_by_uid"],
                            )
                            resolved_candidates.append(
                                {
                                    "label": label_x,
                                    "fen": fen_x,
                                    "eval_override": eval_override_from_ref_meta(meta_x),
                                }
                            )
                        except Exception:
                            continue

                    for profile_name, profile_cfg in profile_map.items():
                        profile_encoder = PositionEncoder(profile_cfg)
                        q_vec_p, _ = encode_board_for_profile(
                            query_board,
                            encoder=profile_encoder,
                            stockfish_path=stockfish_path,
                            eval_depth=max(8, engine_depth),
                            use_engine_eval=bool(demo_use_engine_eval),
                            eval_override=query_eval_override,
                        )
                        temp_rows: list[dict[str, Any]] = []
                        for cand in resolved_candidates:
                            cand_vec_p, _ = encode_board_for_profile(
                                chess.Board(str(cand["fen"])),
                                encoder=profile_encoder,
                                stockfish_path=stockfish_path,
                                eval_depth=max(8, engine_depth),
                                use_engine_eval=bool(demo_use_engine_eval),
                                eval_override=cand.get("eval_override"),
                            )
                            pair = explain_pair(q_vec_p, cand_vec_p, group_slices=profile_encoder.feature_slices())
                            temp_rows.append(
                                {
                                    "profile": profile_name,
                                    "candidate": str(cand["label"]),
                                    "score_100": float(pair["score_100"]),
                                    "distance": float(pair["distance"]),
                                }
                            )
                        temp_rows.sort(key=lambda r: r["score_100"], reverse=True)
                        for idx, row in enumerate(temp_rows, start=1):
                            row["rank"] = idx
                            rank_rows.append(row)
                    if rank_rows:
                        st.dataframe(pd.DataFrame(rank_rows), use_container_width=True)

                st.markdown("**Demo PGN Browser**")
                demo_game_ids = [
                    gid for gid, g in demo_bundle["games_by_uid"].items()
                    if g.source_pgn_file == Path(DEMO_GAMES_PATH).name
                ]
                if demo_game_ids:
                    chosen_demo_game_id = st.selectbox(
                        "Demo PGN game",
                        demo_game_ids,
                        format_func=lambda gid: demo_bundle["games_by_uid"][gid].title(),
                        key="demo_game_select",
                    )
                    chosen_demo_game = demo_bundle["games_by_uid"][chosen_demo_game_id]
                    ply_idx = st.slider(
                        "Demo PGN ply",
                        min_value=0,
                        max_value=len(chosen_demo_game.moves_uci),
                        value=0,
                        key="demo_game_ply",
                    )
                    pgn_board = game_board_at_ply(chosen_demo_game, ply_idx)
                    cg_board(
                        pgn_board,
                        key="demo_pgn_board",
                        orientation=orientation,
                        height=360,
                        disabled=True,
                        last_move_uci=(chosen_demo_game.moves_uci[ply_idx - 1] if ply_idx > 0 else None),
                        arrows=[],
                        instance_id=0,
                    )
                    if st.button("Load Demo PGN Position Into Main Board", use_container_width=True, key="load_demo_game_to_main"):
                        st.session_state.main_tree = AnalysisTree.from_mainline(
                            start_fen=chosen_demo_game.start_fen,
                            moves_uci=chosen_demo_game.moves_uci,
                            headers=chosen_demo_game.headers,
                            current_ply=ply_idx,
                        )
                        st.session_state.main_game_id = chosen_demo_game.game_uid
                        clear_main_restore_stack()
                        bump_main_board_instance()
                        sync_fen_input_to_main()
                        set_notice("success", f"Loaded demo PGN position (ply {ply_idx}) into main board.")
                        st.rerun()
                else:
                    st.caption("No demo PGN games found in the bundled demo dataset.")

                with st.expander("Demo Scenario Recorder / Preset Authoring", expanded=False):
                    st.caption("Capture current boards or scenario boards directly into the bundled demo JSON files.")
                    source_options: dict[str, dict[str, Any]] = {
                        "Current main board": {
                            "fen": st.session_state.main_tree.current_fen,
                            "ref": {"fen": st.session_state.main_tree.current_fen},
                        },
                        "Scenario query": {
                            "fen": query_board.fen(en_passant="fen"),
                            "ref": dict(selected_scenario.get("query", {"fen": query_board.fen(en_passant="fen")})),
                        },
                        "Scenario candidate": {
                            "fen": candidate_board.fen(en_passant="fen"),
                            "ref": dict(selected_scenario.get("candidate", {"fen": candidate_board.fen(en_passant="fen")})),
                        },
                    }
                    if st.session_state.get("similar_tree") is not None:
                        similar_tree_for_save: AnalysisTree = st.session_state.similar_tree
                        source_options["Current similar viewer"] = {
                            "fen": similar_tree_for_save.current_fen,
                            "ref": {"fen": similar_tree_for_save.current_fen},
                        }

                    pos_col_a, pos_col_b = st.columns(2, gap="medium")
                    with pos_col_a:
                        position_source = st.selectbox("Position source", list(source_options.keys()), key="demo_author_position_source")
                        position_id = st.text_input("Position id", value=f"{selected_scenario_id}_capture", key="demo_author_position_id")
                        position_name = st.text_input("Position name", value=f"{selected_scenario.get('title', selected_scenario_id)} capture", key="demo_author_position_name")
                    with pos_col_b:
                        position_description = st.text_area(
                            "Position description",
                            value="Captured from the live demo workspace.",
                            height=90,
                            key="demo_author_position_description",
                        )
                        if st.button("Save Demo Position", use_container_width=True, key="save_demo_position"):
                            saved_position = upsert_demo_position(
                                DEMO_POSITIONS_PATH,
                                position_id=position_id,
                                name=position_name,
                                fen=str(source_options[position_source]["fen"]),
                                description=position_description,
                            )
                            load_demo_positions_cached.clear()
                            build_demo_dataset_cached.clear()
                            set_notice("success", f"Saved demo position `{saved_position['id']}`.")
                            st.rerun()

                    st.markdown("**Save or update a scenario**")
                    scenario_col_a, scenario_col_b = st.columns(2, gap="medium")
                    with scenario_col_a:
                        scenario_id_input = st.text_input("Scenario id", value=f"{selected_scenario_id}_variant", key="demo_author_scenario_id")
                        scenario_title_input = st.text_input(
                            "Scenario title",
                            value=f"{selected_scenario.get('title', selected_scenario_id)} variant",
                            key="demo_author_scenario_title",
                        )
                        scenario_query_source = st.selectbox(
                            "Scenario query source",
                            list(source_options.keys()),
                            index=1 if "Scenario query" in source_options else 0,
                            key="demo_author_query_source",
                        )
                        scenario_candidate_source = st.selectbox(
                            "Scenario candidate source",
                            list(source_options.keys()),
                            index=2 if "Scenario candidate" in source_options else 0,
                            key="demo_author_candidate_source",
                        )
                    with scenario_col_b:
                        scenario_expected_band = st.selectbox(
                            "Expected band",
                            ["perfect", "high", "medium", "low"],
                            index=["perfect", "high", "medium", "low"].index(expected_band) if expected_band in {"perfect", "high", "medium", "low"} else 1,
                            key="demo_author_expected_band",
                        )
                        scenario_feature_focus = st.multiselect(
                            "Feature focus",
                            ["base_781", "piece_counts", "piece_info", "material", "eval"],
                            default=list(selected_scenario.get("feature_focus", [])),
                            key="demo_author_feature_focus",
                        )
                        scenario_narrative_input = st.text_area(
                            "Narrative",
                            value=str(selected_scenario.get("narrative", "")),
                            height=90,
                            key="demo_author_narrative",
                        )
                        if st.button("Save Demo Scenario", use_container_width=True, key="save_demo_scenario"):
                            saved_scenario = upsert_demo_scenario(
                                DEMO_SCENARIOS_PATH,
                                scenario_id=scenario_id_input,
                                title=scenario_title_input,
                                query_ref=dict(source_options[scenario_query_source]["ref"]),
                                candidate_ref=dict(source_options[scenario_candidate_source]["ref"]),
                                expected_band=scenario_expected_band,
                                feature_focus=list(scenario_feature_focus),
                                narrative=scenario_narrative_input,
                            )
                            load_demo_scenarios_cached.clear()
                            set_notice("success", f"Saved demo scenario `{saved_scenario['id']}`.")
                            st.rerun()

    with tabs[2]:
        st.subheader("Benchmark Search Methods")
        st.caption("Measures build time, query latency (p50/p95/p99), estimated index footprint, and optional recall@k vs brute.")
        st.caption("This public benchmark view is intentionally lightweight so the hosted app stays responsive.")
        st.caption(
            "Active corpus source: "
            + ("Bundled bulk PGNs" if corpus_source_mode == "bundled_bulk" else "curated sample PGNs")
        )

        bench_col_a, bench_col_b = st.columns([0.9, 1.1], gap="large")
        with bench_col_a:
            bench_methods = st.multiselect(
                "Methods to benchmark",
                method_options_for_ui(),
                default=["brute", "faiss_flat"] if faiss_available() else ["brute"],
            )
            bench_metric = st.selectbox("Benchmark metric", ["cosine", "l2"], index=0, key="bench_metric")
            bench_k = st.slider("Benchmark k", min_value=1, max_value=20, value=5)
            bench_query_count = st.slider(
                "Sample query count",
                min_value=5,
                max_value=PUBLIC_MAX_BENCH_QUERY_COUNT if PUBLIC_APP_MODE else 200,
                value=PUBLIC_DEFAULT_BENCH_QUERY_COUNT if PUBLIC_APP_MODE else 30,
                step=5,
            )
            bench_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
            bench_compute_recall = st.checkbox("Compute recall@k vs brute", value=True)
            run_bench = st.button("Run Benchmark", type="primary", use_container_width=True)

        with bench_col_b:
            st.markdown("**Benchmark Notes**")
            st.caption("- `brute`: exact NumPy baseline (cosine/L2).")
            st.caption("- `faiss_flat`: exact search in FAISS.")
            st.caption("- `faiss_hnsw` / `faiss_ivf` / `faiss_ivfpq`: approximate FAISS methods.")
            st.caption("- Results use the current sidebar corpus sampling + embedding feature settings.")
            st.caption("- Public hosting uses bundled datasets only; heavy offline benchmark scripts remain available outside the app.")

        if run_bench and not bench_methods:
            st.warning("Select at least one method to benchmark.")
        elif run_bench:
            try:
                enc_cfg_json = json.dumps(encoder_cfg.to_dict(), sort_keys=True)
                corpus_bundle = build_corpus_for_source(
                    corpus_source_mode=corpus_source_mode,
                    pgn_folder_paths=pgn_folder_paths,
                    encoder_cfg_json=enc_cfg_json,
                    every_n_plies=corpus_every_n,
                    min_ply=corpus_min_ply,
                    max_positions=int(corpus_max_positions),
                    dedupe_games=corpus_dedupe_games,
                    dedupe_fen=corpus_dedupe_fen,
                    use_engine_eval_for_corpus=bool(corpus_use_engine_eval),
                    stockfish_path=stockfish_path,
                    eval_depth=max(6, engine_depth - 2),
                )
                bench_method_configs: dict[str, dict[str, Any]] = {
                    "faiss_hnsw": {
                        "M": int(hnsw_m),
                        "ef_construction": int(hnsw_ef_construction),
                        "ef_search": int(hnsw_ef_search),
                    },
                    "faiss_ivf": {"nlist": int(ivf_nlist), "nprobe": int(ivf_nprobe)},
                    "faiss_ivfpq": {
                        "nlist": int(ivf_nlist),
                        "nprobe": int(ivf_nprobe),
                        "pq_m": int(ivfpq_m),
                        "pq_nbits": int(ivfpq_nbits),
                    },
                }
                report = run_search_benchmark(
                    corpus_bundle.vectors,
                    methods=bench_methods,
                    metric=bench_metric,
                    k=bench_k,
                    query_count=bench_query_count,
                    seed=int(bench_seed),
                    method_configs=bench_method_configs,
                    compute_recall=bench_compute_recall,
                )
                report["corpus_manifest"] = corpus_bundle.manifest
                report["corpus_source_mode"] = corpus_source_mode
                st.session_state.last_benchmark_report = report
                st.success("Benchmark completed.")
            except Exception as exc:
                st.error(f"Benchmark failed: {type(exc).__name__}: {exc}")

        report = st.session_state.get("last_benchmark_report")
        if report:
            run_meta = {
                "dataset": report.get("dataset", {}),
                "config": report.get("config", {}),
                "corpus_source_mode": report.get("corpus_source_mode", "curated_samples"),
                "corpus_manifest": report.get("corpus_manifest", {}),
            }
            with st.expander("Benchmark run metadata (JSON)", expanded=True):
                st.code(json.dumps(run_meta, indent=2), language="json")

            results_df = pd.DataFrame(report.get("results", []))
            st.dataframe(results_df, use_container_width=True)

            if not results_df.empty:
                error_col = results_df.get("error", pd.Series([None] * len(results_df), index=results_df.index))
                valid_df = results_df[~error_col.notna()].copy()
                if not valid_df.empty:
                    if "memory_bytes_est" in valid_df.columns:
                        valid_df["memory_mb_est"] = valid_df["memory_bytes_est"].fillna(0.0) / (1024.0 * 1024.0)

                    if "build_time_ms" in valid_df.columns:
                        st.caption("Legend: colors represent search methods. Lower is better for latency/build/memory; higher is better for QPS/recall.")

                    def _render_metric_bar(value_col: str, title: str, y_title: str) -> None:
                        plot_df = valid_df[["method", value_col]].dropna().copy()
                        if plot_df.empty:
                            return
                        chart = (
                            alt.Chart(plot_df)
                            .mark_bar(size=18, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                            .encode(
                                x=alt.X("method:N", sort="-y", title="Search Method", axis=alt.Axis(labelAngle=-18)),
                                y=alt.Y(f"{value_col}:Q", title=y_title),
                                color=alt.Color("method:N", title="Search Method"),
                                tooltip=[
                                    alt.Tooltip("method:N", title="Method"),
                                    alt.Tooltip(f"{value_col}:Q", title=y_title, format=".4f"),
                                ],
                            )
                            .properties(height=260, title=title)
                        )
                        st.altair_chart(styled_benchmark_chart(chart), use_container_width=True)

                    if "memory_mb_est" in valid_df.columns:
                        _render_metric_bar("memory_mb_est", "Estimated Index Footprint by Method", "Estimated Index Footprint (MB)")
                    if "build_time_ms" in valid_df.columns:
                        _render_metric_bar("build_time_ms", "Index Build Time by Method", "Build Time (ms)")
                    if "query_qps_est" in valid_df.columns:
                        _render_metric_bar("query_qps_est", "Estimated Query Throughput", "QPS")
                    if "recall_at_k" in valid_df.columns and valid_df["recall_at_k"].notna().any():
                        _render_metric_bar("recall_at_k", "Retrieval Quality", "Recall@k")

                    latency_cols = [c for c in ("query_latency_ms_p50", "query_latency_ms_p95", "query_latency_ms_p99") if c in valid_df.columns]
                    if latency_cols:
                        latency_df = valid_df[["method"] + latency_cols].melt(
                            id_vars=["method"],
                            var_name="percentile",
                            value_name="latency_ms",
                        )
                        latency_df["percentile"] = latency_df["percentile"].map(
                            {
                                "query_latency_ms_p50": "P50",
                                "query_latency_ms_p95": "P95",
                                "query_latency_ms_p99": "P99",
                            }
                        )
                        latency_df = latency_df.dropna(subset=["latency_ms"])
                        if not latency_df.empty:
                            line = (
                                alt.Chart(latency_df)
                                .mark_line(strokeWidth=2)
                                .encode(
                                    x=alt.X("percentile:N", sort=["P50", "P95", "P99"], title="Latency Percentile"),
                                    y=alt.Y("latency_ms:Q", title="Latency (ms)"),
                                    color=alt.Color("method:N", title="Search Method"),
                                    tooltip=[
                                        alt.Tooltip("method:N", title="Method"),
                                        alt.Tooltip("percentile:N", title="Percentile"),
                                        alt.Tooltip("latency_ms:Q", title="Latency (ms)", format=".4f"),
                                    ],
                                )
                            )
                            pts = (
                                alt.Chart(latency_df)
                                .mark_circle(size=55)
                                .encode(
                                    x=alt.X("percentile:N", sort=["P50", "P95", "P99"]),
                                    y=alt.Y("latency_ms:Q"),
                                    color=alt.Color("method:N", legend=None),
                                )
                            )
                            st.altair_chart(
                                styled_benchmark_chart((line + pts).properties(height=280, title="Latency Profile by Method")),
                                use_container_width=True,
                            )

                    insight_lines: list[str] = []
                    if "query_latency_ms_p50" in valid_df.columns:
                        fastest = valid_df.loc[valid_df["query_latency_ms_p50"].idxmin()]
                        insight_lines.append(f"Fastest p50 latency: `{fastest['method']}` at {fastest['query_latency_ms_p50']:.3f} ms")
                    if "recall_at_k" in valid_df.columns and valid_df["recall_at_k"].notna().any():
                        best_recall = valid_df.loc[valid_df["recall_at_k"].fillna(-1).idxmax()]
                        insight_lines.append(f"Highest recall@k: `{best_recall['method']}` at {float(best_recall['recall_at_k']):.3f}")
                    if insight_lines:
                        st.markdown("**Quick Insights**")
                        for line in insight_lines:
                            st.caption(f"- {line}")


if __name__ == "__main__":
    main()
