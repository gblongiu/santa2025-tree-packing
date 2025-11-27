"""
Hexagonal lattice baseline packer for the Santa 2025 Christmas Tree Packing project.

This module provides a **simple, unique baseline** layout:

- Tree centers lie on a hexagonal (triangular) lattice, which is a natural
  dense packing for circular footprints.
- Tree orientations follow a deterministic pattern based on lattice indices
  (no random angles), so results are fully reproducible given the same
  parameters.

It is intentionally minimal and easy to reason about, and is designed to be
used as the starting point for more advanced methods (e.g. local search).
"""

from __future__ import annotations

from typing import List, Tuple

import math

from ..geometry import TREE_RADIUS, TreePose
from ..config import N_MAX_TREES

# A hex lattice center is represented as (x, y, row, col)
HexCenter = Tuple[float, float, int, int]


# ---------------------------------------------------------------------------
# Hex lattice generation
# ---------------------------------------------------------------------------

def _generate_hex_centers(num_points: int, radius: float = TREE_RADIUS) -> List[HexCenter]:
    """
    Generate a list of hexagonal lattice centers around the origin.

    The layout is constructed by:

    - Using horizontal spacing dx = 2 * radius.
    - Using vertical spacing dy = sqrt(3) * radius.
    - Offsetting every odd row by dx / 2 to create the hex pattern.
    - Enumerating rows and columns in a large square patch around the origin.
    - Sorting all candidates by distance to the origin.
    - Returning the closest `num_points` centers.

    The resulting centers are deterministic and symmetric.
    """
    dx = 2.0 * radius
    dy = math.sqrt(3.0) * radius

    # Generous patch; we trim by distance to origin.
    max_ring = 20
    centers: List[HexCenter] = []

    for row in range(-max_ring, max_ring + 1):
        offset_x = 0.5 * dx if (row & 1) else 0.0
        for col in range(-max_ring, max_ring + 1):
            x = col * dx + offset_x
            y = row * dy
            centers.append((x, y, row, col))

    # Sort by squared distance to origin to keep things stable and deterministic.
    centers.sort(key=lambda c: c[0] * c[0] + c[1] * c[1])
    return centers[:num_points]


# Precompute centers for all n = 1..N_MAX_TREES
_HEX_CENTERS: List[HexCenter] = _generate_hex_centers(N_MAX_TREES)


def get_hex_centers(max_n: int = N_MAX_TREES) -> List[HexCenter]:
    """
    Return the first `max_n` hex centers (closest to the origin).

    This is a convenience wrapper around the precomputed lattice.

    Parameters
    ----------
    max_n:
        Number of centers requested. Must be <= N_MAX_TREES.

    Returns
    -------
    List[HexCenter]
        A shallow copy of the prefix of the global lattice list.
    """
    if max_n < 1 or max_n > len(_HEX_CENTERS):
        raise ValueError(f"max_n must be in [1, {len(_HEX_CENTERS)}], got {max_n}")
    return _HEX_CENTERS[:max_n].copy()


# ---------------------------------------------------------------------------
# Orientation pattern
# ---------------------------------------------------------------------------

def _orientation_for_cell(row: int, col: int, mode: str = "checker") -> float:
    """
    Determine the orientation angle (in degrees) for a tree at a given lattice cell.

    The `mode` argument gives you a hook to experiment with different
    orientation patterns without changing the caller code.

    Modes
    -----
    - 'checker' (default):
        Alternating 0 / 180 degrees based on (row + col) parity.
    - 'row':
        Flip every row: even rows 0 degrees, odd rows 180 degrees.
    - 'radial':
        Orient the tree roughly pointing away from the origin based on its
        lattice coordinates (row, col), for a mild swirl effect.
    """
    if mode == "checker":
        return 0.0 if ((row + col) % 2 == 0) else 180.0
    elif mode == "row":
        return 0.0 if (row % 2 == 0) else 180.0
    elif mode == "radial":
        # Use (row, col) as an approximate direction from origin,
        # then map to angle. This is not exact geometry, just a pattern.
        angle = math.degrees(math.atan2(row, col if col != 0 else 1e-9))
        return angle
    else:
        # Fallback: no rotation
        return 0.0


# ---------------------------------------------------------------------------
# Public API: build baseline layout for a given n
# ---------------------------------------------------------------------------

def initial_hex_layout_for_n(
    n: int,
    orientation_mode: str = "checker",
) -> List[TreePose]:
    """
    Build the hex-lattice baseline layout for a single puzzle with `n` trees.

    - Tree centers are taken from the precomputed hex lattice `_HEX_CENTERS`,
      using the `n` closest centers to the origin.
    - Tree orientations follow a deterministic pattern based on (row, col)
      and the chosen `orientation_mode`.

    Parameters
    ----------
    n:
        Number of trees in this puzzle (1..N_MAX_TREES).
    orientation_mode:
        Orientation pattern to use (see `_orientation_for_cell` for options).

    Returns
    -------
    List[TreePose]
        One `TreePose` per tree, indices implicitly 0..n-1.
    """
    if n < 1 or n > N_MAX_TREES:
        raise ValueError(f"n must be in [1, {N_MAX_TREES}], got {n}")

    centers = get_hex_centers(n)
    poses: List[TreePose] = []

    for (x, y, row, col) in centers:
        angle = _orientation_for_cell(row, col, mode=orientation_mode)
        poses.append(TreePose.from_params(x, y, angle))

    return poses


__all__ = [
    "HexCenter",
    "get_hex_centers",
    "initial_hex_layout_for_n",
]
