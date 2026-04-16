from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import chess
import chess.pgn


@dataclass(slots=True)
class AnalysisNode:
    node_id: str
    fen: str
    parent_id: str | None
    move_uci: str | None
    move_san: str | None
    ply_index: int
    source: str
    children: list[str] = field(default_factory=list)
    main_child_id: str | None = None


@dataclass(slots=True)
class PlayMoveResult:
    accepted: bool
    reason: str = ""
    node_id: str | None = None
    created_branch: bool = False
    reused_existing: bool = False


class AnalysisTree:
    """Branching analysis tree for a chess game/position.

    - Root node stores the starting position (no incoming move)
    - Each child stores the resulting FEN after applying its incoming move
    - `main_child_id` marks the preserved main line at every node
    """

    def __init__(self, root_fen: str = chess.STARTING_FEN, *, headers: dict[str, str] | None = None) -> None:
        self.headers: dict[str, str] = dict(headers or {})
        self.nodes: dict[str, AnalysisNode] = {}
        self._node_counter = 0
        root_id = self._new_node_id()
        self.root_id = root_id
        self.current_id = root_id
        self.nodes[root_id] = AnalysisNode(
            node_id=root_id,
            fen=chess.Board(root_fen).fen(en_passant="fen"),
            parent_id=None,
            move_uci=None,
            move_san=None,
            ply_index=0,
            source="root",
        )
        self.headers.setdefault("Event", "Analysis")
        self.headers.setdefault("Result", "*")

    def clone(self) -> "AnalysisTree":
        return copy.deepcopy(self)

    def _new_node_id(self) -> str:
        node_id = f"n{self._node_counter}"
        self._node_counter += 1
        return node_id

    @property
    def root(self) -> AnalysisNode:
        return self.nodes[self.root_id]

    @property
    def current(self) -> AnalysisNode:
        return self.nodes[self.current_id]

    @property
    def current_fen(self) -> str:
        return self.current.fen

    @property
    def current_ply_index(self) -> int:
        return self.current.ply_index

    def current_board(self) -> chess.Board:
        return chess.Board(self.current.fen)

    def board_for_node(self, node_id: str) -> chess.Board:
        return chess.Board(self.nodes[node_id].fen)

    def reset_to_fen(self, fen: str, *, headers: dict[str, str] | None = None) -> None:
        fresh = AnalysisTree(fen, headers=headers or self.headers)
        self.__dict__.update(fresh.__dict__)

    def path_to_node(self, node_id: str) -> list[str]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        path: list[str] = []
        cur = node_id
        while cur is not None:
            path.append(cur)
            cur = self.nodes[cur].parent_id
        path.reverse()
        return path

    def path_to_current(self) -> list[str]:
        return self.path_to_node(self.current_id)

    def current_line_san(self) -> list[str]:
        out: list[str] = []
        for nid in self.path_to_current()[1:]:
            san = self.nodes[nid].move_san
            if san:
                out.append(san)
        return out

    def can_step_back(self) -> bool:
        return self.current.parent_id is not None

    def can_step_forward(self) -> bool:
        return bool(self.current.children)

    def step_back(self) -> bool:
        parent_id = self.current.parent_id
        if parent_id is None:
            return False
        self.current_id = parent_id
        return True

    def step_forward(self, child_id: str | None = None) -> bool:
        node = self.current
        if not node.children:
            return False
        target = child_id or node.main_child_id or node.children[0]
        if target not in node.children:
            return False
        self.current_id = target
        return True

    def jump_to_start(self) -> None:
        self.current_id = self.root_id

    def jump_to_end(self) -> None:
        # Follow the current variation; at each node prefer its main child.
        while self.step_forward():
            pass

    def set_current(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        self.current_id = node_id
        return True

    def set_current_to_ply_on_mainline(self, ply_index: int) -> None:
        self.jump_to_start()
        target = max(0, int(ply_index))
        while self.current.ply_index < target and self.current.main_child_id is not None:
            self.current_id = self.current.main_child_id

    def _find_child_by_uci(self, parent: AnalysisNode, uci: str) -> str | None:
        for child_id in parent.children:
            if self.nodes[child_id].move_uci == uci:
                return child_id
        return None

    def play_move_uci(self, uci: str, *, allow_branching: bool = True, source: str = "user") -> PlayMoveResult:
        parent = self.current
        board = self.current_board()
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return PlayMoveResult(False, reason=f"Invalid UCI move: {uci}")
        if move not in board.legal_moves:
            return PlayMoveResult(False, reason=f"Illegal move in current position: {uci}")

        existing = self._find_child_by_uci(parent, uci)
        if existing is not None:
            self.current_id = existing
            return PlayMoveResult(True, node_id=existing, reused_existing=True)

        if parent.children and not allow_branching:
            return PlayMoveResult(
                False,
                reason="Branching is disabled in strict game mode. Step forward on the existing line or enable analysis mode.",
            )

        san = board.san(move)
        board.push(move)
        child_id = self._new_node_id()
        child = AnalysisNode(
            node_id=child_id,
            fen=board.fen(en_passant="fen"),
            parent_id=parent.node_id,
            move_uci=uci,
            move_san=san,
            ply_index=parent.ply_index + 1,
            source=source,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)

        created_branch = bool(parent.main_child_id is not None and parent.main_child_id != child_id)
        if parent.main_child_id is None:
            parent.main_child_id = child_id
        self.current_id = child_id

        return PlayMoveResult(True, node_id=child_id, created_branch=created_branch)

    def variations_from_current(self) -> list[dict[str, Any]]:
        node = self.current
        rows: list[dict[str, Any]] = []
        for child_id in node.children:
            child = self.nodes[child_id]
            rows.append(
                {
                    "node_id": child_id,
                    "label": child.move_san or child.move_uci or child_id,
                    "move_uci": child.move_uci,
                    "move_san": child.move_san,
                    "is_mainline": child_id == node.main_child_id,
                    "ply_index": child.ply_index,
                }
            )
        return rows

    def sibling_variations(self) -> list[dict[str, Any]]:
        cur = self.current
        if cur.parent_id is None:
            return []
        parent = self.nodes[cur.parent_id]
        rows: list[dict[str, Any]] = []
        for child_id in parent.children:
            child = self.nodes[child_id]
            rows.append(
                {
                    "node_id": child_id,
                    "label": child.move_san or child.move_uci or child_id,
                    "move_uci": child.move_uci,
                    "move_san": child.move_san,
                    "is_mainline": child_id == parent.main_child_id,
                    "is_current": child_id == cur.node_id,
                    "ply_index": child.ply_index,
                }
            )
        return rows

    def mainline_moves_uci(self) -> list[str]:
        out: list[str] = []
        node = self.root
        while node.main_child_id:
            node = self.nodes[node.main_child_id]
            if node.move_uci:
                out.append(node.move_uci)
        return out

    def ordered_children(self, node_id: str) -> list[str]:
        node = self.nodes[node_id]
        if not node.children:
            return []
        if node.main_child_id is None:
            return list(node.children)
        ordered = [node.main_child_id]
        ordered.extend([cid for cid in node.children if cid != node.main_child_id])
        return ordered

    def replace_with_mainline(
        self,
        *,
        start_fen: str,
        moves_uci: list[str],
        headers: dict[str, str] | None = None,
        source: str = "pgn",
        current_ply: int = 0,
    ) -> None:
        self.reset_to_fen(start_fen, headers=headers or self.headers)
        self.headers.update(headers or {})
        self.jump_to_start()
        for uci in moves_uci:
            result = self.play_move_uci(uci, allow_branching=True, source=source)
            if not result.accepted:
                break
            # Preserve imported line as the main line.
            parent_id = self.nodes[self.current_id].parent_id
            if parent_id is not None:
                self.nodes[parent_id].main_child_id = self.current_id
        self.set_current_to_ply_on_mainline(current_ply)

    @classmethod
    def from_mainline(
        cls,
        *,
        start_fen: str = chess.STARTING_FEN,
        moves_uci: list[str] | None = None,
        headers: dict[str, str] | None = None,
        current_ply: int = 0,
    ) -> "AnalysisTree":
        tree = cls(start_fen, headers=headers)
        if moves_uci:
            tree.replace_with_mainline(
                start_fen=start_fen,
                moves_uci=moves_uci,
                headers=headers,
                source="pgn",
                current_ply=current_ply,
            )
        else:
            tree.set_current_to_ply_on_mainline(current_ply)
        return tree

    def to_pgn_game(self) -> chess.pgn.Game:
        game = chess.pgn.Game()
        for k, v in self.headers.items():
            game.headers[str(k)] = str(v)

        root_board = chess.Board(self.root.fen)
        if self.root.fen != chess.STARTING_FEN:
            game.setup(root_board.copy(stack=False))
            game.headers["FEN"] = root_board.fen(en_passant="fen")
            game.headers["SetUp"] = "1"

        def walk(tree_node_id: str, pgn_node: chess.pgn.GameNode, board: chess.Board) -> None:
            for child_id in self.ordered_children(tree_node_id):
                child = self.nodes[child_id]
                if not child.move_uci:
                    continue
                move = chess.Move.from_uci(child.move_uci)
                if move not in board.legal_moves:
                    continue
                child_pgn_node = pgn_node.add_variation(move)
                next_board = board.copy(stack=False)
                next_board.push(move)
                walk(child_id, child_pgn_node, next_board)

        walk(self.root_id, game, root_board)
        return game

    def to_pgn_string(self) -> str:
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        return self.to_pgn_game().accept(exporter)

