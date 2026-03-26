from __future__ import annotations

from typing import Iterable

Coord = tuple[int, int]
HEX_DIRECTIONS: tuple[Coord, ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))


def ring(center_q: int, center_r: int, radius: int) -> set[Coord]:
    if radius == 0:
        return {(center_q, center_r)}
    results: set[Coord] = set()
    q = center_q + HEX_DIRECTIONS[4][0] * radius
    r = center_r + HEX_DIRECTIONS[4][1] * radius
    for direction in HEX_DIRECTIONS:
        dq, dr = direction
        for _ in range(radius):
            results.add((q, r))
            q += dq
            r += dr
    return results


def neighbors(center_q: int, center_r: int, radius: int = 1) -> set[Coord]:
    results: set[Coord] = set()
    for current_radius in range(radius + 1):
        results.update(ring(center_q, center_r, current_radius))
    return results


def legal_placement_zone(all_hexes: Iterable[Coord], is_first_move: bool) -> set[Coord]:
    occupied = set(all_hexes)
    if is_first_move:
        return neighbors(0, 0, radius=8) - occupied

    zone: set[Coord] = set()
    for q, r in occupied:
        zone.update(neighbors(q, r, radius=8))
    return zone - occupied


def centroid(coords: Iterable[Coord]) -> tuple[float, float]:
    coord_list = list(coords)
    if not coord_list:
        return (0.0, 0.0)
    q_mean = sum(q for q, _ in coord_list) / len(coord_list)
    r_mean = sum(r for _, r in coord_list) / len(coord_list)
    return (q_mean, r_mean)
