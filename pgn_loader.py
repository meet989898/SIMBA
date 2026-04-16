from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import chess
import chess.pgn
import numpy as np

from encoder import PositionEncoder


EvalProvider = Callable[[chess.Board], tuple[int | None, int | None] | None]


@dataclass(slots=True)
class GameRecord:
    game_uid: str
    source_pgn_file: str
    source_path: str
    game_index: int
    headers: dict[str, str]
    start_fen: str
    moves_uci: list[str]
    moves_san: list[str]
    result: str

    def title(self) -> str:
        white = self.headers.get("White", "White")
        black = self.headers.get("Black", "Black")
        event = self.headers.get("Event", "")
        return f"{white} vs {black}" + (f" ({event})" if event else "")


@dataclass(slots=True)
class CorpusBundle:
    vectors: np.ndarray
    positions: list[dict[str, Any]]
    games_by_uid: dict[str, GameRecord]
    manifest: dict[str, Any]


@dataclass(slots=True)
class VectorCorpusBundle:
    vectors: np.ndarray
    manifest: dict[str, Any]


PositionVisitor = Callable[[chess.Board, GameRecord, int], None]


def classify_position_phase(board: chess.Board, ply_index: int) -> str:
    """Lightweight phase classifier for demo/result filtering."""

    non_pawn_material = 0
    queen_count = 0
    for piece_type, value in (
        (chess.KNIGHT, 3),
        (chess.BISHOP, 3),
        (chess.ROOK, 5),
        (chess.QUEEN, 9),
    ):
        count = len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
        if piece_type == chess.QUEEN:
            queen_count = count
        non_pawn_material += count * value

    if non_pawn_material <= 12 or (queen_count == 0 and non_pawn_material <= 18):
        return "endgame"
    if int(ply_index) <= 20:
        return "opening"
    return "middlegame"


def list_sample_pgn_files(sample_dir: str | Path) -> list[Path]:
    return list_pgn_files(sample_dir, recursive=False)


def list_pgn_files(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    pattern = "**/*.pgn" if recursive else "*.pgn"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def iter_pgn_games(
    pgn_path: str | Path,
    *,
    max_games: int | None = None,
) -> Iterator[GameRecord]:
    """Stream games from a PGN file (mainline only), safe for large files on Windows.

    This is the chosen "Option B" pipeline approach: stream/scan multiple PGNs and
    build the combined corpus in memory without writing a merged file.
    """

    pgn_path = Path(pgn_path)
    game_idx = 0
    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            if max_games is not None and game_idx >= max_games:
                break
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            board = game.board()
            start_fen = board.fen(en_passant="fen")
            moves_uci: list[str] = []
            moves_san: list[str] = []
            for move in game.mainline_moves():
                moves_san.append(board.san(move))
                moves_uci.append(move.uci())
                board.push(move)

            headers = {str(k): str(v) for k, v in dict(game.headers).items()}
            uid = f"{pgn_path.name}::game_{game_idx}"
            yield GameRecord(
                game_uid=uid,
                source_pgn_file=pgn_path.name,
                source_path=str(pgn_path),
                game_index=game_idx,
                headers=headers,
                start_fen=start_fen,
                moves_uci=moves_uci,
                moves_san=moves_san,
                result=headers.get("Result", "*"),
            )
            game_idx += 1


def load_games_from_paths(
    pgn_paths: Iterable[str | Path],
    *,
    max_games_per_file: int | None = None,
) -> dict[str, GameRecord]:
    games: dict[str, GameRecord] = {}
    for path in pgn_paths:
        for rec in iter_pgn_games(path, max_games=max_games_per_file):
            games[rec.game_uid] = rec
    return games


def load_curated_sample_games(sample_dir: str | Path) -> dict[str, GameRecord]:
    return load_games_from_paths(list_sample_pgn_files(sample_dir))


def game_board_at_ply(game: GameRecord, ply_index: int) -> chess.Board:
    board = chess.Board(game.start_fen)
    ply_index = max(0, min(int(ply_index), len(game.moves_uci)))
    for uci in game.moves_uci[:ply_index]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Invalid move '{uci}' while reconstructing {game.game_uid}")
        board.push(move)
    return board


def game_pgn_text(game: GameRecord) -> str:
    pgn_game = chess.pgn.Game()
    for k, v in game.headers.items():
        pgn_game.headers[k] = v
    root_board = chess.Board(game.start_fen)
    if game.start_fen != chess.STARTING_FEN:
        pgn_game.setup(root_board.copy(stack=False))
        pgn_game.headers["FEN"] = root_board.fen(en_passant="fen")
        pgn_game.headers["SetUp"] = "1"

    node: chess.pgn.GameNode = pgn_game
    board = root_board.copy(stack=False)
    for uci in game.moves_uci:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        node = node.add_variation(move)
        board.push(move)
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    return pgn_game.accept(exporter)


def scan_pgn_positions(
    pgn_paths: Iterable[str | Path],
    *,
    every_n_plies: int = 2,
    min_ply: int = 1,
    max_ply: int | None = None,
    max_games_per_file: int | None = None,
    max_positions: int | None = None,
    dedupe_games: bool = False,
    dedupe_fen: bool = False,
    include_start_position: bool = False,
    visitor: PositionVisitor | None = None,
) -> dict[str, Any]:
    """Stream corpus positions without materializing per-position metadata.

    This is intended for offline benchmark/caching workflows where we need
    deterministic corpus traversal but do not need the full in-memory app bundle.
    """

    pgn_path_list = [Path(p) for p in pgn_paths]
    seen_fens: set[str] = set()
    seen_game_signatures: set[str] = set()
    duplicate_games_skipped = 0
    total_games = 0
    positions_loaded = 0
    unique_games_kept = 0

    for pgn_path in pgn_path_list:
        for game in iter_pgn_games(pgn_path, max_games=max_games_per_file):
            total_games += 1
            if dedupe_games:
                signature_raw = f"{game.start_fen}|{' '.join(game.moves_uci)}"
                signature = hashlib.sha1(signature_raw.encode("utf-8", errors="ignore")).hexdigest()
                if signature in seen_game_signatures:
                    duplicate_games_skipped += 1
                    continue
                seen_game_signatures.add(signature)
            unique_games_kept += 1

            board = chess.Board(game.start_fen)
            if include_start_position and min_ply <= 0:
                fen0 = board.fen(en_passant="fen")
                if (not dedupe_fen) or (fen0 not in seen_fens):
                    if dedupe_fen:
                        seen_fens.add(fen0)
                    if visitor is not None:
                        visitor(board, game, 0)
                    positions_loaded += 1
                    if max_positions is not None and positions_loaded >= max_positions:
                        return {
                            "pipeline": "stream_scan_positions",
                            "pgn_files": [str(p) for p in pgn_path_list],
                            "games_loaded": total_games,
                            "positions_loaded": positions_loaded,
                            "every_n_plies": every_n_plies,
                            "min_ply": min_ply,
                            "max_ply": max_ply,
                            "dedupe_games": dedupe_games,
                            "dedupe_fen": dedupe_fen,
                            "duplicate_games_skipped": duplicate_games_skipped,
                            "unique_games_kept": unique_games_kept,
                        }

            for ply, uci in enumerate(game.moves_uci, start=1):
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    break
                board.push(move)

                if ply < min_ply:
                    continue
                if max_ply is not None and ply > max_ply:
                    break
                if every_n_plies > 1 and (ply % every_n_plies) != 0:
                    continue

                fen = board.fen(en_passant="fen")
                if dedupe_fen:
                    if fen in seen_fens:
                        continue
                    seen_fens.add(fen)

                if visitor is not None:
                    visitor(board, game, ply)
                positions_loaded += 1
                if max_positions is not None and positions_loaded >= max_positions:
                    return {
                        "pipeline": "stream_scan_positions",
                        "pgn_files": [str(p) for p in pgn_path_list],
                        "games_loaded": total_games,
                        "positions_loaded": positions_loaded,
                        "every_n_plies": every_n_plies,
                        "min_ply": min_ply,
                        "max_ply": max_ply,
                        "dedupe_games": dedupe_games,
                        "dedupe_fen": dedupe_fen,
                        "duplicate_games_skipped": duplicate_games_skipped,
                        "unique_games_kept": unique_games_kept,
                    }

    return {
        "pipeline": "stream_scan_positions",
        "pgn_files": [str(p) for p in pgn_path_list],
        "games_loaded": total_games,
        "positions_loaded": positions_loaded,
        "every_n_plies": every_n_plies,
        "min_ply": min_ply,
        "max_ply": max_ply,
        "dedupe_games": dedupe_games,
        "dedupe_fen": dedupe_fen,
        "duplicate_games_skipped": duplicate_games_skipped,
        "unique_games_kept": unique_games_kept,
    }


def build_position_corpus_in_memory(
    pgn_paths: Iterable[str | Path],
    encoder: PositionEncoder,
    *,
    every_n_plies: int = 2,
    min_ply: int = 1,
    max_ply: int | None = None,
    max_games_per_file: int | None = None,
    max_positions: int | None = None,
    dedupe_games: bool = False,
    dedupe_fen: bool = False,
    include_start_position: bool = False,
    eval_provider: EvalProvider | None = None,
) -> CorpusBundle:
    """Scan multiple PGNs and build a combined in-memory corpus (no merged file written).

    This avoids an extra giant merged PGN on Windows and keeps the pipeline streaming-friendly.
    """

    pgn_path_list = [Path(p) for p in pgn_paths]
    vectors: list[np.ndarray] = []
    positions: list[dict[str, Any]] = []
    games_by_uid: dict[str, GameRecord] = {}
    seen_fens: set[str] = set()
    seen_game_signatures: set[str] = set()
    duplicate_games_skipped = 0

    total_games = 0
    for pgn_path in pgn_path_list:
        for game in iter_pgn_games(pgn_path, max_games=max_games_per_file):
            total_games += 1
            if dedupe_games:
                signature_raw = f"{game.start_fen}|{' '.join(game.moves_uci)}"
                signature = hashlib.sha1(signature_raw.encode("utf-8", errors="ignore")).hexdigest()
                if signature in seen_game_signatures:
                    duplicate_games_skipped += 1
                    continue
                seen_game_signatures.add(signature)
            games_by_uid[game.game_uid] = game

            board = chess.Board(game.start_fen)
            if include_start_position and min_ply <= 0:
                fen0 = board.fen(en_passant="fen")
                if (not dedupe_fen) or (fen0 not in seen_fens):
                    if dedupe_fen:
                        seen_fens.add(fen0)
                    eval_pair = eval_provider(board) if eval_provider else None
                    cp, mate = eval_pair if eval_pair is not None else (None, None)
                    vectors.append(encoder.encode(board, eval_cp=cp, eval_mate=mate))
                    positions.append(
                        {
                            "corpus_index": len(positions),
                            "source_pgn_file": game.source_pgn_file,
                            "source_path": game.source_path,
                            "game_id": game.game_uid,
                            "game_index": game.game_index,
                            "ply_index": 0,
                            "move_index": -1,
                            "fullmove_number": board.fullmove_number,
                            "turn": "w" if board.turn == chess.WHITE else "b",
                            "phase": classify_position_phase(board, 0),
                            "fen": fen0,
                            "eval_cp_stm": cp,
                            "eval_mate_stm": mate,
                            "result": game.result,
                            "headers": dict(game.headers),
                        }
                    )

            for ply, uci in enumerate(game.moves_uci, start=1):
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    break
                board.push(move)

                if ply < min_ply:
                    continue
                if max_ply is not None and ply > max_ply:
                    break
                if every_n_plies > 1 and (ply % every_n_plies) != 0:
                    continue

                fen = board.fen(en_passant="fen")
                if dedupe_fen:
                    if fen in seen_fens:
                        continue
                    seen_fens.add(fen)

                eval_pair = eval_provider(board) if eval_provider else None
                cp, mate = eval_pair if eval_pair is not None else (None, None)
                vec = encoder.encode(board, eval_cp=cp, eval_mate=mate)
                vectors.append(vec)
                positions.append(
                    {
                        "corpus_index": len(positions),
                        "source_pgn_file": game.source_pgn_file,
                        "source_path": game.source_path,
                        "game_id": game.game_uid,
                        "game_index": game.game_index,
                        "ply_index": ply,
                        "move_index": ply - 1,
                        "fullmove_number": board.fullmove_number,
                        "turn": "w" if board.turn == chess.WHITE else "b",
                        "phase": classify_position_phase(board, ply),
                        "fen": fen,
                        "eval_cp_stm": cp,
                        "eval_mate_stm": mate,
                        "result": game.result,
                        "headers": dict(game.headers),  # required by app + deliverable metadata
                    }
                )

                if max_positions is not None and len(positions) >= max_positions:
                    matrix = np.vstack(vectors).astype(np.float32, copy=False) if vectors else np.zeros((0, encoder.dim), dtype=np.float32)
                    return CorpusBundle(
                        vectors=matrix,
                        positions=positions,
                        games_by_uid=games_by_uid,
                        manifest={
                            "pipeline": "option_b_in_memory_combined_scan",
                            "pgn_files": [str(p) for p in pgn_path_list],
                            "games_loaded": total_games,
                            "positions_loaded": len(positions),
                            "encoder": encoder.config.to_dict(),
                            "encoder_dim": encoder.dim,
                            "every_n_plies": every_n_plies,
                            "min_ply": min_ply,
                            "max_ply": max_ply,
                            "dedupe_games": dedupe_games,
                            "dedupe_fen": dedupe_fen,
                            "duplicate_games_skipped": duplicate_games_skipped,
                            "unique_games_kept": len(games_by_uid),
                        },
                    )

    matrix = np.vstack(vectors).astype(np.float32, copy=False) if vectors else np.zeros((0, encoder.dim), dtype=np.float32)
    return CorpusBundle(
        vectors=matrix,
        positions=positions,
        games_by_uid=games_by_uid,
        manifest={
            "pipeline": "option_b_in_memory_combined_scan",
            "pgn_files": [str(p) for p in pgn_path_list],
            "games_loaded": total_games,
            "positions_loaded": len(positions),
            "encoder": encoder.config.to_dict(),
            "encoder_dim": encoder.dim,
            "every_n_plies": every_n_plies,
            "min_ply": min_ply,
            "max_ply": max_ply,
            "dedupe_games": dedupe_games,
            "dedupe_fen": dedupe_fen,
            "duplicate_games_skipped": duplicate_games_skipped,
            "unique_games_kept": len(games_by_uid),
        },
    )


def validate_pgn_file(pgn_path: str | Path, *, max_games: int = 3) -> tuple[bool, str]:
    try:
        count = 0
        for _ in iter_pgn_games(pgn_path, max_games=max_games):
            count += 1
        if count == 0:
            return False, "No games parsed."
        return True, f"Parsed {count} game(s) successfully."
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
