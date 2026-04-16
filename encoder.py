from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np


PIECE_ORDER = (
    (chess.WHITE, chess.PAWN),
    (chess.WHITE, chess.KNIGHT),
    (chess.WHITE, chess.BISHOP),
    (chess.WHITE, chess.ROOK),
    (chess.WHITE, chess.QUEEN),
    (chess.WHITE, chess.KING),
    (chess.BLACK, chess.PAWN),
    (chess.BLACK, chess.KNIGHT),
    (chess.BLACK, chess.BISHOP),
    (chess.BLACK, chess.ROOK),
    (chess.BLACK, chess.QUEEN),
    (chess.BLACK, chess.KING),
)

PIECE_INDEX = {key: i for i, key in enumerate(PIECE_ORDER)}

PIECE_MAX_COUNTS = {
    chess.PAWN: 8.0,
    chess.KNIGHT: 2.0,
    chess.BISHOP: 2.0,
    chess.ROOK: 2.0,
    chess.QUEEN: 1.0,
    chess.KING: 1.0,
}

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def _normalize_eval(cp: int | None = None, mate: int | None = None) -> float:
    if mate is not None:
        if mate == 0:
            return 0.0
        sign = 1.0 if mate > 0 else -1.0
        return float(sign * (0.95 + 0.05 * (1.0 - min(abs(mate), 20) / 20.0)))
    if cp is None:
        return 0.0
    return float(np.tanh(float(cp) / 600.0))


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    include_eval_score: bool = False
    include_piece_counts: bool = True
    include_piece_info: bool = True
    include_material_balance: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_eval_score": self.include_eval_score,
            "include_piece_counts": self.include_piece_counts,
            "include_piece_info": self.include_piece_info,
            "include_material_balance": self.include_material_balance,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "EncoderConfig":
        raw = raw or {}
        return cls(
            include_eval_score=bool(raw.get("include_eval_score", False)),
            include_piece_counts=bool(raw.get("include_piece_counts", True)),
            include_piece_info=bool(raw.get("include_piece_info", True)),
            include_material_balance=bool(raw.get("include_material_balance", True)),
        )


class PositionEncoder:
    """Deterministic chess position encoder.

    Base representation is a strict 781-d structural encoding:
    - 12 x 64 piece planes = 768
    - side to move = 1
    - castling rights = 4
    - en-passant file one-hot = 8
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()

    @property
    def dim(self) -> int:
        d = 781
        if self.config.include_eval_score:
            d += 1
        if self.config.include_piece_counts:
            d += 12
        if self.config.include_piece_info:
            d += 10
        if self.config.include_material_balance:
            d += 8
        return d

    def feature_slices(self) -> dict[str, slice]:
        """Return deterministic vector slice boundaries for active config."""
        i = 0
        slices: dict[str, slice] = {"base_781": slice(i, i + 781)}
        i += 781
        if self.config.include_eval_score:
            slices["eval"] = slice(i, i + 1)
            i += 1
        if self.config.include_piece_counts:
            slices["piece_counts"] = slice(i, i + 12)
            i += 12
        if self.config.include_piece_info:
            slices["piece_info"] = slice(i, i + 10)
            i += 10
        if self.config.include_material_balance:
            slices["material"] = slice(i, i + 8)
            i += 8
        if i != self.dim:
            raise RuntimeError(f"Feature slices cover {i} dims but expected {self.dim}")
        return slices

    def _encode_base_781(self, board: chess.Board, out: np.ndarray, offset: int = 0) -> int:
        for square, piece in board.piece_map().items():
            plane = PIECE_INDEX[(piece.color, piece.piece_type)]
            out[offset + plane * 64 + square] = 1.0
        i = offset + 768
        out[i] = 1.0 if board.turn == chess.WHITE else 0.0
        i += 1

        out[i] = float(board.has_kingside_castling_rights(chess.WHITE))
        out[i + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
        out[i + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
        out[i + 3] = float(board.has_queenside_castling_rights(chess.BLACK))
        i += 4

        # En-passant file one-hot (0s when no ep square)
        if board.ep_square is not None:
            out[i + chess.square_file(board.ep_square)] = 1.0
        i += 8
        return i

    def _encode_piece_counts(self, board: chess.Board, out: np.ndarray, offset: int) -> int:
        i = offset
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
                count = len(board.pieces(piece_type, color))
                out[i] = float(count) / PIECE_MAX_COUNTS[piece_type]
                i += 1
        return i

    def _encode_piece_info(self, board: chess.Board, out: np.ndarray, offset: int) -> int:
        i = offset
        for color in (chess.WHITE, chess.BLACK):
            sq = board.king(color)
            if sq is None:
                out[i] = 0.0
                out[i + 1] = 0.0
            else:
                out[i] = chess.square_file(sq) / 7.0
                out[i + 1] = chess.square_rank(sq) / 7.0
            i += 2

        out[i] = min(float(board.legal_moves.count()), 218.0) / 218.0
        out[i + 1] = float(board.is_check())
        out[i + 2] = float(board.is_checkmate())
        out[i + 3] = float(board.is_stalemate())
        out[i + 4] = float(board.is_insufficient_material())
        out[i + 5] = float(board.can_claim_fifty_moves())
        return i + 6

    def _encode_material_balance(self, board: chess.Board, out: np.ndarray, offset: int) -> int:
        i = offset
        white_total = 0.0
        black_total = 0.0
        balances: list[float] = []
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            w = len(board.pieces(piece_type, chess.WHITE))
            b = len(board.pieces(piece_type, chess.BLACK))
            value = PIECE_VALUES[piece_type]
            white_total += w * value
            black_total += b * value
            max_balance = PIECE_MAX_COUNTS[piece_type] * value
            balances.append(((w - b) * value) / max_balance if max_balance else 0.0)

        out[i] = white_total / 39.0
        out[i + 1] = black_total / 39.0
        out[i + 2] = (white_total - black_total) / 39.0
        i += 3
        for val in balances:
            out[i] = float(val)
            i += 1
        return i

    def encode(
        self,
        board: chess.Board,
        *,
        eval_cp: int | None = None,
        eval_mate: int | None = None,
    ) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        i = self._encode_base_781(board, vec, 0)

        if self.config.include_eval_score:
            vec[i] = _normalize_eval(cp=eval_cp, mate=eval_mate)
            i += 1
        if self.config.include_piece_counts:
            i = self._encode_piece_counts(board, vec, i)
        if self.config.include_piece_info:
            i = self._encode_piece_info(board, vec, i)
        if self.config.include_material_balance:
            i = self._encode_material_balance(board, vec, i)

        if i != self.dim:
            raise RuntimeError(f"Encoder wrote {i} dims but expected {self.dim}")
        return vec

    def encode_fen(
        self,
        fen: str,
        *,
        eval_cp: int | None = None,
        eval_mate: int | None = None,
    ) -> np.ndarray:
        return self.encode(chess.Board(fen), eval_cp=eval_cp, eval_mate=eval_mate)


def validate_fen(fen: str) -> tuple[bool, str]:
    try:
        chess.Board(fen)
        return True, ""
    except ValueError as exc:
        return False, str(exc)
