from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import streamlit.components.v1 as components


_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

_cg_component = components.declare_component(
    "capstone_chessground_component",
    path=str(_FRONTEND_DIR),
)


def _ensure_fen(board_or_fen: chess.Board | str) -> str:
    if isinstance(board_or_fen, chess.Board):
        return board_or_fen.fen(en_passant="fen")
    return str(board_or_fen)


def cg_board(
    board_or_fen: chess.Board | str,
    *,
    key: str,
    orientation: str = "white",
    height: int = 560,
    disabled: bool = False,
    last_move_uci: str | None = None,
    arrows: list[dict[str, str]] | None = None,
    instance_id: int = 0,
) -> dict[str, Any] | None:
    """Render the Chessground Streamlit custom component.

    Returns a move event payload emitted by the frontend (or ``None``):
      {
        "fen": "...",
        "prev_fen": "...",
        "uci": "e2e4",
        "san": "e4",
        "seq": 1,
        "instance_id": 0,
        "kind": "move" | "size",
        "board_px": 560
      }
    """

    return _cg_component(
        fen=_ensure_fen(board_or_fen),
        orientation="black" if orientation == "black" else "white",
        height=int(height),
        disabled=bool(disabled),
        last_move_uci=(last_move_uci or ""),
        arrows=arrows or [],
        instance_id=int(instance_id),
        key=key,
        default=None,
    )
