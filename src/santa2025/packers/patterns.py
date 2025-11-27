-"""
Experimental patterns and tilings for the Santa 2025 Christmas Tree Packing project.

This module is a playground for generating **parameterized patterns** of tree
centers and orientations, independent of any particular competition metric.

The idea is to separate:
- How we choose the *centers* (grid, rotated grid, concentric rings, etc).
- How we choose the *orientation* of each tree at those centers.

You can then:
- Turn these patterns into `TreePose` layouts.
- Compare them against the hex or radial baselines.
- Feed them into local search as alternative starting points.
"""

from __future__ import annotations

from typing import List, Tuple
import math

import numpy as np

from ..geometry import TreePose, TREE_RADIUS

Point2D = Tuple[float, float]


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def orientation_from_center(
    x: float,
    y: float,
    idx: int,
    mode: str = "fixed",
    base_angle_deg: float = 0.0,
) -> float:
    """
    Decide the tree orientation at a given center point.

    Modes
    -----
    - 'fixed' (default):
        Use a single constant angle, `base_angle_deg`.
    - 'alternate':
        Alternate between `base_angle_deg` and `base_angle_deg + 180` by index.
    - 'outward':
        Orient the tree roughly pointing away from the origin, plus `base_angle_deg`.
    - 'inward':
        Orient the tree roughly pointing toward the origin, plus `base_angle_deg`.
    - 'tangent':
        Orient along a tangent direction relative to the origin, producing a swirl.
    """
    mode = mode.lower()

    if mode == "fixed":
        return base_angle_deg

    if mode == "alternate":
        return base_angle_deg if (idx % 2 == 0) else (base_angle_deg + 180.0)

    # For the radial styles we need the polar angle of the center.
    angle_from_origin = math.degrees(math.atan2(y, x)) if (x != 0.0 or y != 0.0) else 0.0

    if mode == "outward":
        return angle_from_origin + base_angle_deg

    if mode == "inward":
        return angle_from_origin + base_angle_deg + 180.0

    if mode == "tangent":
        # Perpendicular to radius vector, simple swirl.
        return angle_from_origin + 90.0 + base_angle_deg

    # Fallback
    return base_angle_deg


# ---------------------------------------------------------------------------
# Center patterns
# ---------------------------------------------------------------------------

def regular_grid_centers(
    rows: int,
    cols: int,
    spacing: float | None = None,
) -> List[Point2D]:
    """
    Regular axis-aligned grid centered at the origin.

    Parameters
    ----------
    rows, cols:
        Number of rows and columns.
    spacing:
        Distance between neighboring centers along x and y. If None,
        uses 1.8 * TREE_RADIUS as a loosely non-overlapping value.

    Returns
    -------
    List[Point2D]
        List of (x, y) centers.
    """
    if spacing is None:
        spacing = 1.8 * TREE_RADIUS

    xs = np.arange(cols) - (cols - 1) / 2.0
    ys = np.arange(rows) - (rows - 1) / 2.0

    centers: List[Point2D] = []
    for j, y_idx in enumerate(ys):
        for i, x_idx in enumerate(xs):
            centers.append((float(x_idx * spacing), float(y_idx * spacing)))
    return centers


def rotated_grid_centers(
    count: int,
    spacing: float | None = None,
    angle_deg: float = 45.0,
) -> List[Point2D]:
    """
    Rotated grid pattern trimmed to the closest `count` points by radius.

    Implementation:
    - Start from a large square grid around the origin.
    - Rotate all grid points by `angle_deg` around the origin.
    - Sort by distance to origin and keep the first `count`.

    Parameters
    ----------
    count:
        Number of centers to return.
    spacing:
        Grid spacing before rotation. If None, uses 1.8 * TREE_RADIUS.
    angle_deg:
        Rotation angle for the grid.

    Returns
    -------
    List[Point2D]
        List of (x, y) centers.
    """
    if spacing is None:
        spacing = 1.8 * TREE_RADIUS

    # Heuristic grid size large enough to cover `count` points after trimming.
    side = max(3, int(math.ceil(math.sqrt(count)) * 2))
    xs = np.arange(-side, side + 1) * spacing
    ys = np.arange(-side, side + 1) * spacing

    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    centers: List[Point2D] = []
    for y in ys:
        for x in xs:
            rx = c * x - s * y
            ry = s * x + c * y
            centers.append((float(rx), float(ry)))

    centers.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return centers[:count]


def concentric_rings_centers(
    num_rings: int,
    base_points_per_ring: int = 6,
    ring_step: float | None = None,
) -> List[Point2D]:
    """
    Concentric rings pattern around the origin.

    Ring k (1-based) has:
    - radius r_k = k * ring_step
    - points = base_points_per_ring * k

    Parameters
    ----------
    num_rings:
        Number of rings to generate.
    base_points_per_ring:
        Number of points on the first ring; subsequent rings scale linearly.
    ring_step:
        Radial step between rings. If None, uses 2.0 * TREE_RADIUS.

    Returns
    -------
    List[Point2D]
        List of (x, y) centers, excluding the origin.
    """
    if ring_step is None:
        ring_step = 2.0 * TREE_RADIUS

    centers: List[Point2D] = []
    for k in range(1, num_rings + 1):
        radius = k * ring_step
        num_points = base_points_per_ring * k
        for i in range(num_points):
            angle = 2.0 * math.pi * i / float(num_points)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            centers.append((float(x), float(y)))
    return centers


# ---------------------------------------------------------------------------
# Layout builders from centers
# ---------------------------------------------------------------------------

def layout_from_centers(
    centers: List[Point2D],
    orientation_mode: str = "fixed",
    base_angle_deg: float = 0.0,
) -> List[TreePose]:
    """
    Turn a list of centers into a list of `TreePose` objects.

    This is a generic utility that applies an orientation pattern to each
    center and calls `TreePose.from_params`.

    Parameters
    ----------
    centers:
        List of (x, y) positions.
    orientation_mode:
        Orientation pattern; see `orientation_from_center` for options.
    base_angle_deg:
        Base angle that some modes add or alternate with.

    Returns
    -------
    List[TreePose]
        Tree poses corresponding to the desired pattern.
    """
    poses: List[TreePose] = []
    for idx, (x, y) in enumerate(centers):
        angle = orientation_from_center(x, y, idx, mode=orientation_mode, base_angle_deg=base_angle_deg)
        poses.append(TreePose.from_params(x, y, angle))
    return poses


def grid_layout(
    rows: int,
    cols: int,
    spacing: float | None = None,
    orientation_mode: str = "fixed",
    base_angle_deg: float = 0.0,
) -> List[TreePose]:
    """
    Convenience wrapper for a regular grid layout.

    Returns
    -------
    List[TreePose]
        Grid-based layout with the requested orientation pattern.
    """
    centers = regular_grid_centers(rows=rows, cols=cols, spacing=spacing)
    return layout_from_centers(centers, orientation_mode=orientation_mode, base_angle_deg=base_angle_deg)


def rotated_grid_layout(
    count: int,
    spacing: float | None = None,
    angle_deg: float = 45.0,
    orientation_mode: str = "fixed",
    base_angle_deg: float = 0.0,
) -> List[TreePose]:
    """
    Convenience wrapper for a rotated grid layout with trimming.

    Parameters
    ----------
    count:
        Number of trees desired.
    spacing, angle_deg, orientation_mode, base_angle_deg:
        Passed through to `rotated_grid_centers` and `layout_from_centers`.

    Returns
    -------
    List[TreePose]
        Rotated-grid layout with the given orientation pattern.
    """
    centers = rotated_grid_centers(count=count, spacing=spacing, angle_deg=angle_deg)
    return layout_from_centers(centers, orientation_mode=orientation_mode, base_angle_deg=base_angle_deg)


def concentric_rings_layout(
    num_rings: int,
    base_points_per_ring: int = 6,
    ring_step: float | None = None,
    orientation_mode: str = "outward",
    base_angle_deg: float = 0.0,
    include_center: bool = True,
) -> List[TreePose]:
    """
    Convenience wrapper for a concentric rings layout.

    Parameters
    ----------
    num_rings, base_points_per_ring, ring_step:
        Passed through to `concentric_rings_centers`.
    orientation_mode, base_angle_deg:
        Passed through to `layout_from_centers`.
    include_center:
        If True, add a single tree at the origin before the rings.

    Returns
    -------
    List[TreePose]
        Ring-based layout with the chosen orientation pattern.
    """
    centers: List[Point2D] = []
    if include_center:
        centers.append((0.0, 0.0))

    centers.extend(concentric_rings_centers(
        num_rings=num_rings,
        base_points_per_ring=base_points_per_ring,
        ring_step=ring_step,
    ))
    return layout_from_centers(centers, orientation_mode=orientation_mode, base_angle_deg=base_angle_deg)


__all__ = [
    "Point2D",
    "orientation_from_center",
    "regular_grid_centers",
    "rotated_grid_centers",
    "concentric_rings_centers",
    "layout_from_centers",
    "grid_layout",
    "rotated_grid_layout",
    "concentric_rings_layout",
]
