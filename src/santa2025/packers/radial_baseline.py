"""
Radial / greedy baseline packer for the Santa 2025 Christmas Tree Packing project.

This module implements an alternative baseline that is independent from the
hexagonal lattice approach:

- Trees are assigned directions based on a **golden–angle spiral** in polar
  coordinates, which spreads rays evenly around the origin.
- Along each ray, the tree is placed by **greedily increasing the radius**
  until it fits without overlapping any previously placed trees and stays
  inside the allowed coordinate window.
- Tree orientation is controlled by a small set of deterministic patterns
  (outward facing, fixed, alternating).

This packer is mainly useful for comparison and for seeding other algorithms.
"""

from __future__ import annotations

from typing import List
import math

from shapely.strtree import STRtree

from ..geometry import TreePose, make_tree_polygon
from ..config import N_MAX_TREES, COORD_MIN, COORD_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coords_in_bounds(x: float, y: float) -> bool:
    """
    Check that (x, y) is inside the competition coordinate range.
    """
    return (COORD_MIN <= x <= COORD_MAX) and (COORD_MIN <= y <= COORD_MAX)


def _golden_angle_sequence_deg(n: int, start_angle_deg: float = 0.0) -> List[float]:
    """
    Generate n angles in degrees using a golden–angle sequence.

    This creates a well–spread set of directions on [0, 360).
    """
    # Golden angle in degrees: 360 * (1 - 1/phi)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    golden_angle = 360.0 * (1.0 - 1.0 / phi)

    angles = []
    angle = start_angle_deg
    for _ in range(n):
        angles.append(angle % 360.0)
        angle += golden_angle
    return angles


def _orientation_for_tree(index: int, dir_angle_deg: float, mode: str) -> float:
    """
    Decide the tree orientation for a given tree index and ray direction.

    Modes
    -----
    - 'outward' (default):
        Angle equals the ray direction (roughly points away from origin).
    - 'fixed':
        All trees use angle 0 degrees.
    - 'alternate':
        Even indices 0 degrees, odd indices 180 degrees.
    """
    mode = mode.lower()
    if mode == "outward":
        return dir_angle_deg
    if mode == "fixed":
        return 0.0
    if mode == "alternate":
        return 0.0 if (index % 2 == 0) else 180.0
    # Fallback
    return dir_angle_deg


def _place_tree_along_ray(
    vx: float,
    vy: float,
    angle_deg: float,
    existing_poses: List[TreePose],
    radial_step: float = 0.15,
    max_radius: float = 90.0,
) -> TreePose:
    """
    Place a single tree along the ray with direction (vx, vy).

    We start at radius 0 and increase in steps of `radial_step` until we find
    a position that:

    - Keeps (x, y) within [COORD_MIN, COORD_MAX].
    - Does not overlap any existing tree polygons (touching is allowed).

    If we exceed `max_radius` without finding a feasible position, a
    RuntimeError is raised.
    """
    # Precompute existing polygons and an index (if any trees already placed).
    existing_polys = [p.poly for p in existing_poses]
    index = STRtree(existing_polys) if existing_polys else None

    radius = 0.0
    while radius <= max_radius:
        x = radius * vx
        y = radius * vy

        if not _coords_in_bounds(x, y):
            radius += radial_step
            continue

        poly = make_tree_polygon(x, y, angle_deg)

        if index is not None:
            cand_indices = index.query(poly)
            collision = any(
                poly.intersects(existing_polys[j]) and not poly.touches(existing_polys[j])
                for j in cand_indices
            )
            if collision:
                radius += radial_step
                continue

        # Found a feasible position
        return TreePose(x=x, y=y, angle=angle_deg, poly=poly)

        radius += radial_step

    raise RuntimeError(
        "Could not place tree along ray without collision within max_radius. "
        "Consider increasing max_radius or adjusting radial_step."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initial_radial_layout_for_n(
    n: int,
    orientation_mode: str = "outward",
    radial_step: float = 0.15,
    max_radius: float = 90.0,
    start_angle_deg: float = 0.0,
) -> List[TreePose]:
    """
    Build a radial / greedy baseline layout for a single puzzle with `n` trees.

    Algorithm outline
    -----------------
    1. Generate a golden–angle sequence of `n` directions.
    2. For each tree index k:
       - Convert the k-th angle to a unit direction vector (vx, vy).
       - Choose its orientation angle based on `orientation_mode`.
       - Place it greedily along the ray by stepping radius outward until
         there is no collision and coordinates stay in range.

    Parameters
    ----------
    n:
        Number of trees in this puzzle (1..N_MAX_TREES).
    orientation_mode:
        Tree orientation pattern ('outward', 'fixed', 'alternate').
    radial_step:
        Radial increment for the greedy search along each ray.
    max_radius:
        Maximum search radius. If placement fails before this radius, a
        RuntimeError is raised.
    start_angle_deg:
        Starting angle for the golden–angle sequence.

    Returns
    -------
    List[TreePose]
        One `TreePose` per tree, indices implicitly 0..n - 1.
    """
    if n < 1 or n > N_MAX_TREES:
        raise ValueError(f"n must be in [1, {N_MAX_TREES}], got {n}")

    angles_dir = _golden_angle_sequence_deg(n, start_angle_deg=start_angle_deg)

    poses: List[TreePose] = []
    for idx, dir_angle_deg in enumerate(angles_dir):
        theta = math.radians(dir_angle_deg)
        vx = math.cos(theta)
        vy = math.sin(theta)

        orient_deg = _orientation_for_tree(idx, dir_angle_deg, mode=orientation_mode)
        pose = _place_tree_along_ray(
            vx=vx,
            vy=vy,
            angle_deg=orient_deg,
            existing_poses=poses,
            radial_step=radial_step,
            max_radius=max_radius,
        )
        poses.append(pose)

    return poses


__all__ = [
    "initial_radial_layout_for_n",
]
