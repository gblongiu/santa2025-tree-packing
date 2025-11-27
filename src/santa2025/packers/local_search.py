"""
Local search refinements for the Santa 2025 Christmas Tree Packing project.

This module implements *geometry-level* improvement steps on top of a baseline
layout (for example the hexagonal lattice from `hex_lattice.py`).

The primary API is:

    refine_layout(poses, iters=..., move_scale=..., angle_scale=...)

which performs a simple hill-climbing / (optional) annealing-style local search:

- At each step, pick one tree at random.
- Propose a small perturbation in position (dx, dy) and angle (d_angle).
- Reject moves that cause overlaps or go outside coordinate bounds.
- Accept moves that reduce the bounding square side.
- Optionally accept some worse moves with a probability that decays over time
  (simulated annealing flavor).

The refinement operates *in-place* on the given list of `TreePose` instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable
import math
import random

from shapely.strtree import STRtree

from ..geometry import TreePose
from ..config import (
    COORD_MIN,
    COORD_MAX,
    LOCAL_SEARCH_MOVE_SCALE,
    LOCAL_SEARCH_ANGLE_SCALE,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def bounding_square_side(poses: List[TreePose]) -> float:
    """
    Compute the side length of the smallest axis aligned square that
    contains all polygons in `poses`.

    This is similar in spirit to `evaluation.bounding_square_side_for_group`,
    but operates directly on a list of `TreePose` objects.

    Implementation note
    -------------------
    The naive version used shapely.ops.unary_union(...) which becomes
    expensive when called thousands of times in local search loops.

    Because valid layouts never have overlapping trees, the bounding box
    of the union is identical to the bounding box of the individual
    polygons. So we can just aggregate bounds in O(N) without building
    any unions.
    """
    if not poses:
        return 0.0

    polys = [p.poly for p in poses]

    # Each .bounds is (minx, miny, maxx, maxy)
    minx = min(g.bounds[0] for g in polys)
    miny = min(g.bounds[1] for g in polys)
    maxx = max(g.bounds[2] for g in polys)
    maxy = max(g.bounds[3] for g in polys)

    width = maxx - minx
    height = maxy - miny
    return float(max(width, height))


def has_any_collision(poses: List[TreePose]) -> bool:
    """
    Check whether any pair of polygons in `poses` has a *true* overlap.

    Trees that only touch along edges or points are considered OK
    (no collision) – this matches the competition's notion of overlap.
    """
    polys = [p.poly for p in poses]
    index = STRtree(polys)
    for i, poly in enumerate(polys):
        candidates = index.query(poly)
        for j in candidates:
            if i == j:
                continue
            if poly.intersects(polys[j]) and not poly.touches(polys[j]):
                return True
    return False


def _coords_in_bounds(x: float, y: float) -> bool:
    """
    Check that (x, y) lies inside the allowed coordinate range.
    """
    return (COORD_MIN <= x <= COORD_MAX) and (COORD_MIN <= y <= COORD_MAX)


# ---------------------------------------------------------------------------
# Local search / hill-climbing core
# ---------------------------------------------------------------------------

@dataclass
class LocalSearchStats:
    """
    Simple statistics object summarizing a local search run.
    """

    initial_side: float
    final_side: float
    accepted_moves: int
    rejected_moves: int
    improved_moves: int

    @property
    def improvement(self) -> float:
        return self.initial_side - self.final_side


def _default_temperature_schedule(
    step: int,
    iters: int,
    temp_start: float,
    temp_end: float,
) -> float:
    """
    Linear temperature schedule from `temp_start` down to `temp_end`.

    If both are zero, behaves like pure hill climbing (no uphill moves).
    """
    if iters <= 1:
        return temp_end
    alpha = step / float(iters - 1)
    return (1.0 - alpha) * temp_start + alpha * temp_end


def refine_layout(
    poses: List[TreePose],
    iters: int = 2_000,
    move_scale: Optional[float] = None,
    angle_scale: Optional[float] = None,
    allow_annealing: bool = False,
    temp_start: float = 0.1,
    temp_end: float = 0.001,
    temperature_schedule: Optional[
        Callable[[int, int, float, float], float]
    ] = None,
    track_stats: bool = False,
) -> float | LocalSearchStats:
    """
    Refine a list of tree poses via hill climbing / simulated annealing.

    The refinement is performed *in-place* on `poses`.

    Parameters
    ----------
    poses:
        List of TreePose objects representing an initial non-overlapping layout.
    iters:
        Number of local-search iterations to perform.
    move_scale:
        Standard deviation (in world units) for proposed x/y moves.
        If None, uses `LOCAL_SEARCH_MOVE_SCALE` from config.
    angle_scale:
        Standard deviation (in degrees) for proposed angle changes.
        If None, uses `LOCAL_SEARCH_ANGLE_SCALE` from config.
    allow_annealing:
        If True, may accept some worse moves with a probability that decays
        from `temp_start` to `temp_end` via `temperature_schedule`.
    temp_start, temp_end:
        Starting / ending temperatures for the schedule if `allow_annealing`
        is True. Units are in "score delta", i.e. we use exp(-delta / T).
    temperature_schedule:
        Optional custom schedule function with signature
            f(step, iters, temp_start, temp_end) -> temperature
        If None, a simple linear schedule is used.
    track_stats:
        If True, returns a `LocalSearchStats` object instead of just the
        final best side length.

    Returns
    -------
    float or LocalSearchStats
        If track_stats is False: the final best side length.
        If track_stats is True: a LocalSearchStats summary.

    Notes
    -----
    - This function assumes the initial layout has no overlaps.
    - It preserves the "no overlaps" invariant by rejecting any move
      that introduces a collision or violates coordinate bounds.
    """
    if not poses:
        return LocalSearchStats(0.0, 0.0, 0, 0, 0) if track_stats else 0.0

    move_scale = LOCAL_SEARCH_MOVE_SCALE if move_scale is None else move_scale
    angle_scale = LOCAL_SEARCH_ANGLE_SCALE if angle_scale is None else angle_scale

    temp_fn = temperature_schedule or _default_temperature_schedule

    # Initial objective
    best_side = bounding_square_side(poses)
    initial_side = best_side

    accepted = 0
    rejected = 0
    improved = 0

    # Main loop
    for step in range(iters):
        # Choose a random tree to perturb
        idx = random.randrange(len(poses))
        tree = poses[idx]

        old_x, old_y, old_angle = tree.x, tree.y, tree.angle
        old_poly = tree.poly

        # Propose a small random move (Gaussian step)
        dx = move_scale * random.gauss(0.0, 1.0)
        dy = move_scale * random.gauss(0.0, 1.0)
        d_angle = angle_scale * random.gauss(0.0, 1.0)

        new_x = old_x + dx
        new_y = old_y + dy
        new_angle = old_angle + d_angle

        # Enforce coordinate bounds up front
        if not _coords_in_bounds(new_x, new_y):
            rejected += 1
            continue

        # Apply candidate pose
        tree.x, tree.y, tree.angle = new_x, new_y, new_angle
        tree.update()

        # Collision check only against other trees
        others = [p.poly for j, p in enumerate(poses) if j != idx]
        index = STRtree(others)
        cand_indices = index.query(tree.poly)

        collision = any(
            tree.poly.intersects(others[j]) and not tree.poly.touches(others[j])
            for j in cand_indices
        )

        if collision:
            # Revert and reject
            tree.x, tree.y, tree.angle = old_x, old_y, old_angle
            tree.poly = old_poly
            rejected += 1
            continue

        # Compute new objective
        new_side = bounding_square_side(poses)
        delta = new_side - best_side

        accept_move = False
        if delta <= 0.0:
            # Improvement (or tie)
            accept_move = True
            if delta < 0.0:
                improved += 1
        elif allow_annealing:
            # Possibly accept a worse move according to temperature
            T = temp_fn(step, iters, temp_start, temp_end)
            if T > 0.0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept_move = True

        if accept_move:
            best_side = new_side
            accepted += 1
        else:
            # Revert pose
            tree.x, tree.y, tree.angle = old_x, old_y, old_angle
            tree.poly = old_poly
            rejected += 1

    if track_stats:
        return LocalSearchStats(
            initial_side=initial_side,
            final_side=best_side,
            accepted_moves=accepted,
            rejected_moves=rejected,
            improved_moves=improved,
        )

    return best_side


__all__ = [
    "bounding_square_side",
    "has_any_collision",
    "LocalSearchStats",
    "refine_layout",
]
