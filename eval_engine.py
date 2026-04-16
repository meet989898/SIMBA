from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

import chess
import chess.engine
import streamlit as st


DEFAULT_STOCKFISH_PATH = r"C:\tools\stockfish\stockfish.exe"
MATE_CP_PROXY = 100_000


@dataclass(slots=True)
class EngineEval:
    ok: bool
    error: str | None = None
    depth: int | None = None
    cp_stm: int | None = None
    mate_stm: int | None = None
    normalized_stm: float = 0.0
    normalized_white: float = 0.0
    eval_bar_white: float = 0.5
    numeric_stm: str = "0.00"
    best_move_uci: str | None = None
    pv_uci: list[str] = field(default_factory=list)
    pv_san: list[str] = field(default_factory=list)


def normalize_eval_for_embedding(cp: int | None = None, mate: int | None = None) -> float:
    """Map engine scores into [-1, 1] for embeddings/UI.

    Positive values favor the side-to-move (when using ``cp_stm``/``mate_stm``).
    """

    if mate is not None:
        if mate == 0:
            return 0.0
        sign = 1.0 if mate > 0 else -1.0
        # Closer mates saturate near +/-1.0.
        proximity = 1.0 - min(abs(mate), 20) / 20.0
        return float(sign * (0.95 + 0.05 * proximity))
    if cp is None:
        return 0.0
    return float(math.tanh(float(cp) / 600.0))


def eval_bar_fraction_from_normalized_white(norm_white: float) -> float:
    """Convert normalized white-perspective eval [-1,1] -> [0,1]."""

    return max(0.0, min(1.0, 0.5 + (norm_white * 0.5)))


def format_eval_numeric(cp: int | None = None, mate: int | None = None) -> str:
    if mate is not None:
        if mate > 0:
            return f"#{mate}"
        return f"#-{abs(mate)}"
    if cp is None:
        return "0.00"
    return f"{cp / 100.0:+.2f}"


def _pv_to_san(board: chess.Board, pv: list[chess.Move], max_moves: int = 12) -> list[str]:
    tmp = board.copy(stack=False)
    out: list[str] = []
    for move in pv[:max_moves]:
        if move not in tmp.legal_moves:
            break
        out.append(tmp.san(move))
        tmp.push(move)
    return out


@st.cache_resource(show_spinner=False)
def get_engine(stockfish_path: str) -> chess.engine.SimpleEngine:
    # Cache by path so changing the sidebar path cleanly creates a new process.
    if sys.platform.startswith("win"):
        # python-chess engine startup can fail with NotImplementedError on Windows
        # when Streamlit runs in a non-main thread with a non-Proactor loop policy.
        proactor_policy_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
        if proactor_policy_cls is not None and not isinstance(asyncio.get_event_loop_policy(), proactor_policy_cls):
            asyncio.set_event_loop_policy(proactor_policy_cls())
    return chess.engine.SimpleEngine.popen_uci(stockfish_path)


def analyze_position(
    board: chess.Board,
    *,
    stockfish_path: str,
    depth: int = 12,
    pv_moves: int = 10,
    time_limit_s: float | None = None,
) -> EngineEval:
    """Analyze a position safely and return side-to-move and white-perspective summaries."""

    path = str(Path(stockfish_path))
    if not path.strip():
        return EngineEval(ok=False, error="Stockfish path is empty.")

    try:
        engine = get_engine(path)
    except FileNotFoundError:
        return EngineEval(ok=False, error=f"Stockfish executable not found: {path}")
    except PermissionError:
        return EngineEval(ok=False, error=f"Permission denied when launching Stockfish: {path}")
    except OSError as exc:
        return EngineEval(ok=False, error=f"Unable to start Stockfish ({type(exc).__name__}): {exc}")
    except NotImplementedError as exc:
        return EngineEval(
            ok=False,
            error=(
                "Engine initialization failed (NotImplementedError). "
                "On Windows, ensure the app runs with a Proactor event loop policy. "
                f"Details: {exc}"
            ),
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        return EngineEval(ok=False, error=f"Engine initialization failed ({type(exc).__name__}): {exc}")

    try:
        limit = chess.engine.Limit(depth=int(depth)) if time_limit_s is None else chess.engine.Limit(
            depth=int(depth), time=float(time_limit_s)
        )
        info = engine.analyse(board, limit)
    except chess.engine.EngineTerminatedError:
        # Clear cached dead process and report.
        get_engine.clear()
        return EngineEval(ok=False, error="Stockfish process terminated unexpectedly.")
    except chess.engine.EngineError as exc:
        return EngineEval(ok=False, error=f"Engine error: {exc}")
    except Exception as exc:  # pragma: no cover - environment-specific
        return EngineEval(ok=False, error=f"Analysis failed ({type(exc).__name__}): {exc}")

    pov_score = info.get("score")
    if pov_score is None:
        return EngineEval(ok=False, error="Engine returned no score.")

    score_stm = pov_score.pov(board.turn)
    mate_stm = score_stm.mate()
    cp_stm = score_stm.score(mate_score=MATE_CP_PROXY)
    norm_stm = normalize_eval_for_embedding(cp=cp_stm, mate=mate_stm)
    norm_white = norm_stm if board.turn == chess.WHITE else -norm_stm
    pv = list(info.get("pv", []))
    best_move = pv[0] if pv else None

    return EngineEval(
        ok=True,
        depth=int(info.get("depth") or depth),
        cp_stm=cp_stm,
        mate_stm=mate_stm,
        normalized_stm=norm_stm,
        normalized_white=norm_white,
        eval_bar_white=eval_bar_fraction_from_normalized_white(norm_white),
        numeric_stm=format_eval_numeric(cp_stm, mate_stm),
        best_move_uci=best_move.uci() if best_move else None,
        pv_uci=[m.uci() for m in pv[:pv_moves]],
        pv_san=_pv_to_san(board, pv, max_moves=pv_moves),
    )
