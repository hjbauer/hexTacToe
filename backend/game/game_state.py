from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Coord = tuple[int, int]


@dataclass(frozen=True)
class GameState:
    red_hexes: frozenset[Coord] = field(default_factory=frozenset)
    blue_hexes: frozenset[Coord] = field(default_factory=frozenset)
    turn_number: int = 0
    placements_this_turn: int = 0
    is_terminal: bool = False
    winner: Optional[str] = None
    last_move_hexes: frozenset[Coord] = field(default_factory=frozenset)

    @property
    def current_player(self) -> str:
        return "red" if self.turn_number % 2 == 0 else "blue"

    @property
    def hexes_required_this_turn(self) -> int:
        return 1 if self.turn_number == 0 else 2

    @property
    def all_hexes(self) -> frozenset[Coord]:
        return self.red_hexes | self.blue_hexes

    @property
    def placements_remaining_this_turn(self) -> int:
        return self.hexes_required_this_turn - self.placements_this_turn

    def to_dict(self) -> dict:
        return {
            "red": [list(coord) for coord in sorted(self.red_hexes)],
            "blue": [list(coord) for coord in sorted(self.blue_hexes)],
            "turn_number": self.turn_number,
            "placements_this_turn": self.placements_this_turn,
            "current_player": self.current_player,
            "hexes_required_this_turn": self.hexes_required_this_turn,
            "placements_remaining_this_turn": self.placements_remaining_this_turn,
            "is_terminal": self.is_terminal,
            "winner": self.winner,
            "last_move_hexes": [list(coord) for coord in sorted(self.last_move_hexes)],
        }
