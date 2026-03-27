from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random

from backend.game.game_state import Coord, GameState
from backend.game.hex_coord import HEX_DIRECTIONS, centroid, hex_distance
from backend.game.rules import HEX_AXES, check_win, get_legal_moves


@dataclass(frozen=True)
class TacticalSelectorResult:
    coord: Coord
    policy: dict[Coord, float]
    reason: str
    forced_moves: tuple[Coord, ...]


def _line_length_with_move(state: GameState, player: str, move: Coord) -> int:
    owned = set(state.red_hexes if player == "red" else state.blue_hexes)
    owned.add(move)
    best = 1
    for dq, dr in HEX_AXES:
        count = 1
        nq, nr = move[0] + dq, move[1] + dr
        while (nq, nr) in owned:
            count += 1
            nq += dq
            nr += dr
        nq, nr = move[0] - dq, move[1] - dr
        while (nq, nr) in owned:
            count += 1
            nq -= dq
            nr -= dr
        best = max(best, count)
    return best


def longest_line_for_player(state: GameState, player: str) -> int:
    owned = state.red_hexes if player == "red" else state.blue_hexes
    if not owned:
        return 0
    best = 0
    for move in owned:
        best = max(best, _line_length_with_move(state, player, move))
    return best


def adjacent_friendly_count(state: GameState, player: str, move: Coord) -> int:
    owned = state.red_hexes if player == "red" else state.blue_hexes
    return sum(1 for coord in owned if hex_distance(move[0], move[1], coord[0], coord[1]) == 1)


def would_win_if_played(state: GameState, player: str, move: Coord) -> bool:
    owned = set(state.red_hexes if player == "red" else state.blue_hexes)
    owned.add(move)
    return check_win(frozenset(owned))


def _occupied_by_player(state: GameState, player: str) -> frozenset[Coord]:
    return state.red_hexes if player == "red" else state.blue_hexes


def _threat_segments(state: GameState, player: str) -> list[tuple[Coord, ...]]:
    owned = _occupied_by_player(state, player)
    if not owned:
        return []
    segments: list[tuple[Coord, ...]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for q, r in owned:
        for dq, dr in HEX_AXES:
            for offset in range(6):
                start_q = q - (dq * offset)
                start_r = r - (dr * offset)
                key = (start_q, start_r, dq, dr)
                if key in seen:
                    continue
                segment = tuple(
                    (start_q + (dq * step), start_r + (dr * step))
                    for step in range(6)
                )
                stone_count = sum(1 for coord in segment if coord in owned)
                if stone_count < 4:
                    continue
                seen.add(key)
                segments.append(segment)
    return segments


@lru_cache(maxsize=131072)
def _threat_plans_cached(
    state: GameState,
    player: str,
    max_empty: int = 2,
) -> tuple[tuple[Coord, ...], ...]:
    owned = _occupied_by_player(state, player)
    opponent = "blue" if player == "red" else "red"
    blocked = _occupied_by_player(state, opponent)
    legal_moves = get_legal_moves(state)
    plans: set[tuple[Coord, ...]] = set()

    # A next-turn tactical threat only depends on short contiguous windows
    # on the three hex axes. Scanning those windows avoids expensive legal-pair search.
    for segment in _threat_segments(state, player):
        if any(coord in blocked for coord in segment):
            continue
        empties = tuple(coord for coord in segment if coord not in owned)
        if not empties or len(empties) > max_empty:
            continue
        if any(coord not in legal_moves for coord in empties):
            continue
        plans.add(tuple(sorted(empties)))

    return tuple(sorted(plans))


def _threat_plans(state: GameState, player: str, max_empty: int = 2) -> tuple[tuple[Coord, ...], ...]:
    return _threat_plans_cached(state, player, max_empty)


def immediate_winning_moves(state: GameState, player: str) -> list[Coord]:
    return [plan[0] for plan in _threat_plans(state, player, max_empty=1)]


def next_turn_winning_plans(state: GameState, player: str) -> list[tuple[Coord, ...]]:
    return list(_threat_plans(state, player, max_empty=2))


def winning_moves_this_turn(state: GameState, player: str) -> dict[Coord, float]:
    budget = state.placements_remaining_this_turn
    if budget <= 0:
        return {}
    plans = [plan for plan in next_turn_winning_plans(state, player) if len(plan) <= budget]
    if not plans:
        return {}

    scores = score_legal_moves(state)
    move_scores: dict[Coord, float] = {}
    for move in sorted(get_legal_moves(state)):
        coverage = 0.0
        for plan in plans:
            if move in plan:
                coverage += 10.0 if len(plan) == 1 else 4.0
        if coverage <= 0:
            continue
        move_scores[move] = coverage * 1000.0 + scores.get(move, 0.0)

    if not move_scores:
        return {}
    best = max(move_scores.values())
    exp_scores = {
        move: pow(2.718281828, (score - best) / 16.0)
        for move, score in move_scores.items()
    }
    total = sum(exp_scores.values()) or 1.0
    return {move: value / total for move, value in exp_scores.items() if value > 1e-9}


def connected_components(state: GameState, player: str) -> list[set[Coord]]:
    owned = set(_occupied_by_player(state, player))
    components: list[set[Coord]] = []
    while owned:
        start = owned.pop()
        component = {start}
        stack = [start]
        while stack:
            q, r = stack.pop()
            for dq, dr in HEX_DIRECTIONS:
                neighbor = (q + dq, r + dr)
                if neighbor in owned:
                    owned.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def average_nearest_friendly_distance(state: GameState, player: str) -> float:
    owned = list(_occupied_by_player(state, player))
    if len(owned) < 2:
        return 0.0
    total = 0.0
    for index, coord in enumerate(owned):
        nearest = min(
            hex_distance(coord[0], coord[1], other[0], other[1])
            for other_index, other in enumerate(owned)
            if other_index != index
        )
        total += float(nearest)
    return total / float(len(owned))


def _can_cover_plans(
    chosen: tuple[Coord, ...],
    plans: list[tuple[Coord, ...]],
    remaining_budget: int,
) -> bool:
    uncovered = [plan for plan in plans if not set(chosen).intersection(plan)]
    if not uncovered:
        return True
    if remaining_budget <= 0:
        return False
    candidate_coords = sorted({coord for plan in uncovered for coord in plan if coord not in chosen})
    for coord in candidate_coords:
        if _can_cover_plans(chosen + (coord,), uncovered, remaining_budget - 1):
            return True
    return False


def blocking_moves_for_next_turn(state: GameState, opponent: str) -> dict[Coord, float]:
    defense_budget = state.placements_remaining_this_turn
    if defense_budget <= 0:
        return {}
    plans = _threat_plans(state, opponent, max_empty=2)
    if not plans:
        return {}

    scores = score_legal_moves(state)
    legal_moves = sorted(get_legal_moves(state))
    move_scores: dict[Coord, float] = {}
    fallback_scores: dict[Coord, float] = {}

    for move in legal_moves:
        coverage = 0.0
        for plan in plans:
            if move in plan:
                coverage += 8.0 if len(plan) == 1 else 3.0
        if coverage <= 0:
            continue
        fallback_scores[move] = coverage * 1000.0 + scores.get(move, 0.0)
        if _can_cover_plans((move,), plans, defense_budget - 1):
            move_scores[move] = coverage * 1000.0 + scores.get(move, 0.0)

    active_scores = move_scores if move_scores else fallback_scores
    if not active_scores:
        return {}
    best = max(active_scores.values())
    exp_scores = {
        move: pow(2.718281828, (score - best) / 16.0)
        for move, score in active_scores.items()
    }
    total = sum(exp_scores.values()) or 1.0
    normalized = {move: value / total for move, value in exp_scores.items()}
    return {move: value for move, value in normalized.items() if value > 1e-9}


@lru_cache(maxsize=131072)
def _forced_move_policy_cached(
    state: GameState,
) -> tuple[Coord, tuple[tuple[Coord, float], ...], str, tuple[Coord, ...]] | None:
    legal_moves = sorted(get_legal_moves(state))
    if not legal_moves:
        return None

    player = state.current_player
    opponent = "blue" if player == "red" else "red"
    scores = score_legal_moves(state)

    winning_policy = winning_moves_this_turn(state, player)
    if winning_policy:
        winning_moves = tuple(sorted(move for move, prob in winning_policy.items() if prob > 0.0))
        chosen = max(winning_policy.items(), key=lambda item: (item[1], scores.get(item[0], 0.0), item[0]))[0]
        return (
            chosen,
            tuple(sorted(winning_policy.items())),
            "win_now" if any(len(plan) == 1 for plan in next_turn_winning_plans(state, player)) else "win_this_turn",
            winning_moves,
        )

    blocking_policy = blocking_moves_for_next_turn(state, opponent)
    if blocking_policy:
        blocking_moves = tuple(sorted(move for move, prob in blocking_policy.items() if prob > 0.0))
        chosen = max(blocking_policy.items(), key=lambda item: (item[1], scores.get(item[0], 0.0), item[0]))[0]
        return (
            chosen,
            tuple(sorted(blocking_policy.items())),
            "block_now" if any(len(plan) == 1 for plan in next_turn_winning_plans(state, opponent)) else "block_next_turn",
            blocking_moves,
        )

    return None


def forced_move_policy(state: GameState) -> TacticalSelectorResult | None:
    cached = _forced_move_policy_cached(state)
    if cached is None:
        return None
    coord, policy_items, reason, forced_moves = cached
    return TacticalSelectorResult(
        coord=coord,
        policy=dict(policy_items),
        reason=reason,
        forced_moves=forced_moves,
    )


@lru_cache(maxsize=131072)
def _score_legal_moves_cached(state: GameState) -> tuple[tuple[Coord, float], ...]:
    legal_moves = list(get_legal_moves(state))
    if not legal_moves:
        return ()

    player = state.current_player
    opponent = "blue" if player == "red" else "red"
    all_hexes = list(state.all_hexes)
    center_q, center_r = centroid(all_hexes)
    scored: dict[Coord, float] = {}

    for move in legal_moves:
        if would_win_if_played(state, player, move):
            scored[move] = 1_000_000.0
            continue

        score = 0.0
        if would_win_if_played(state, opponent, move):
            score += 50_000.0

        player_line = _line_length_with_move(state, player, move)
        opponent_line = _line_length_with_move(state, opponent, move)
        score += player_line * player_line * 20.0
        score += opponent_line * opponent_line * 8.0

        friendly_neighbors = adjacent_friendly_count(state, player, move)
        enemy_neighbors = sum(
            1
            for owned in (state.blue_hexes if player == "red" else state.red_hexes)
            if hex_distance(move[0], move[1], owned[0], owned[1]) == 1
        )
        score += friendly_neighbors * 6.0
        score += enemy_neighbors * 3.0
        score -= hex_distance(move[0], move[1], round(center_q), round(center_r)) * 1.5
        score -= hex_distance(move[0], move[1], 0, 0) * 0.5

        scored[move] = score

    return tuple(sorted(scored.items()))


def score_legal_moves(state: GameState) -> dict[Coord, float]:
    return dict(_score_legal_moves_cached(state))


@lru_cache(maxsize=131072)
def _heuristic_policy_cached(state: GameState) -> tuple[tuple[Coord, float], ...]:
    scores = dict(_score_legal_moves_cached(state))
    if not scores:
        return ()
    max_score = max(scores.values())
    exp_scores = {move: pow(2.718281828, (score - max_score) / 12.0) for move, score in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return tuple(sorted((move, value / total) for move, value in exp_scores.items()))


def heuristic_policy(state: GameState) -> dict[Coord, float]:
    return dict(_heuristic_policy_cached(state))


class RandomBaseline:
    name = "random"

    def select_move(self, state: GameState) -> Coord:
        legal_moves = sorted(get_legal_moves(state))
        if not legal_moves:
            raise ValueError("no legal moves available")
        return random.choice(legal_moves)


class HeuristicBaseline:
    name = "heuristic"

    def select_move(self, state: GameState) -> Coord:
        policy = heuristic_policy(state)
        if not policy:
            raise ValueError("no legal moves available")
        return max(policy.items(), key=lambda item: item[1])[0]
