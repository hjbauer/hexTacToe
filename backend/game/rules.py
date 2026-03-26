from __future__ import annotations

from dataclasses import replace

from backend.game.game_state import Coord, GameState
from backend.game.hex_coord import legal_placement_zone

HEX_AXES = [(1, 0), (0, 1), (1, -1)]


def check_win(hexes: frozenset[Coord]) -> bool:
    hex_set = set(hexes)
    for q, r in hexes:
        for dq, dr in HEX_AXES:
            count = 1
            nq, nr = q + dq, r + dr
            while (nq, nr) in hex_set:
                count += 1
                nq += dq
                nr += dr
            nq, nr = q - dq, r - dr
            while (nq, nr) in hex_set:
                count += 1
                nq -= dq
                nr -= dr
            if count >= 6:
                return True
    return False


def get_legal_moves(state: GameState) -> set[Coord]:
    if state.is_terminal:
        return set()
    return legal_placement_zone(state.all_hexes, is_first_move=not state.all_hexes)


def apply_move(state: GameState, coord: Coord) -> GameState:
    if state.is_terminal:
        raise ValueError("game is already terminal")
    if coord in state.all_hexes:
        raise ValueError("hex already occupied")
    if coord not in get_legal_moves(state):
        raise ValueError("hex not in legal zone")

    current_player = state.current_player
    red_hexes = state.red_hexes
    blue_hexes = state.blue_hexes

    if current_player == "red":
        red_hexes = frozenset(set(red_hexes) | {coord})
        active_hexes = red_hexes
    else:
        blue_hexes = frozenset(set(blue_hexes) | {coord})
        active_hexes = blue_hexes

    if state.placements_this_turn == 0:
        new_last_move_hexes = frozenset({coord})
    else:
        new_last_move_hexes = frozenset(set(state.last_move_hexes) | {coord})
    placements_this_turn = state.placements_this_turn + 1

    if check_win(active_hexes):
        return replace(
            state,
            red_hexes=red_hexes,
            blue_hexes=blue_hexes,
            placements_this_turn=placements_this_turn,
            is_terminal=True,
            winner=current_player,
            last_move_hexes=new_last_move_hexes,
        )

    if placements_this_turn >= state.hexes_required_this_turn:
        return replace(
            state,
            red_hexes=red_hexes,
            blue_hexes=blue_hexes,
            turn_number=state.turn_number + 1,
            placements_this_turn=0,
            last_move_hexes=new_last_move_hexes,
        )

    return replace(
        state,
        red_hexes=red_hexes,
        blue_hexes=blue_hexes,
        placements_this_turn=placements_this_turn,
        last_move_hexes=new_last_move_hexes,
    )


def apply_moves(state: GameState, coords: list[Coord]) -> GameState:
    if len(coords) > state.placements_remaining_this_turn:
        raise ValueError("too many placements for current turn")
    updated = state
    for coord in coords:
        updated = apply_move(updated, coord)
        if updated.is_terminal:
            break
    return updated
