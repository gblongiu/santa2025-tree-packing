"""
Geometry utilities for the Santa 2025 – Christmas Tree Packing project.

This module defines:

- The canonical Christmas tree polygon in **local coordinates**.
- Its circumscribed radius (useful for safe spacing / hex packing).
- Helpers to build rotated / translated Shapely polygons.
- A small `TreePose` dataclass representing a placed tree.

Coordinate convention
---------------------

The local coordinate system is defined so that:

- The origin (0, 0) is the **center of the top of the trunk**.
- The tree extends upward to roughly y = 0.8 (the tip).
- The trunk extends downward to y = -0.2.

This matches the competition definition that (x, y) specifies the center
of the top of the trunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import math

import numpy as np
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Tree definition in local coordinates
# ---------------------------------------------------------------------------

# Vertices of the tree polygon in local coordinates (x, y).
# Origin (0, 0) is the center of the top of the trunk.
TREE_TEMPLATE_VERTS: np.ndarray = np.array(
    [
        [0.0, 0.8],          # Tip of the tree
        [0.25 / 2, 0.5],     # Right top tier
        [0.25 / 4, 0.5],
        [0.4 / 2, 0.25],     # Right middle tier
        [0.4 / 4, 0.25],
        [0.7 / 2, 0.0],      # Right bottom tier
        [0.15 / 2, 0.0],     # Right trunk top
        [0.15 / 2, -0.2],    # Right trunk bottom
        [-0.15 / 2, -0.2],   # Left trunk bottom
        [-0.15 / 2, 0.0],    # Left trunk top
        [-0.7 / 2, 0.0],     # Left bottom tier
        [-0.4 / 4, 0.25],    # Left middle tier
        [-0.4 / 2, 0.25],
        [-0.25 / 4, 0.5],    # Left top tier
        [-0.25 / 2, 0.5],
    ],
    dtype=float,
)

#: Circumscribed radius of the tree polygon in local coordinates.
#: This is the maximum distance from the origin to any vertex.
TREE_RADIUS: float = float(np.linalg.norm(TREE_TEMPLATE_VERTS, axis=1).max())


def make_tree_polygon_local() -> Polygon:
    """
    Return the canonical tree polygon in local coordinates as a Shapely Polygon.

    The polygon is defined by `TREE_TEMPLATE_VERTS` and is not rotated or translated.
    """
    return Polygon(TREE_TEMPLATE_VERTS)


def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """
    Construct a Shapely Polygon for a tree placed at (x, y) with rotation `angle_deg`.

    Parameters
    ----------
    x, y:
        Coordinates of the tree in world space (center of top of the trunk).
    angle_deg:
        Rotation angle in degrees, counter-clockwise.

    Returns
    -------
    shapely.geometry.Polygon
        The transformed polygon.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    # 2x2 rotation matrix
    rot = np.array([[c, -s], [s, c]], dtype=float)

    # Apply rotation then translation
    pts = TREE_TEMPLATE_VERTS @ rot.T
    pts[:, 0] += x
    pts[:, 1] += y

    return Polygon(pts)


# ---------------------------------------------------------------------------
# TreePose: a placed tree (pose + polygon)
# ---------------------------------------------------------------------------

@dataclass
class TreePose:
    """
    Represents a single placed tree:

    - (x, y): position of the center of the top of the trunk
    - angle: rotation in degrees
    - poly: corresponding Shapely Polygon (kept in sync by `update()`)

    This is a light, convenient container for local search and packing code.
    """

    x: float
    y: float
    angle: float
    poly: Polygon

    @classmethod
    def from_params(cls, x: float, y: float, angle: float) -> "TreePose":
        """
        Create a TreePose from (x, y, angle) by building the polygon.
        """
        return cls(x=x, y=y, angle=angle, poly=make_tree_polygon(x, y, angle))

    def update(self) -> None:
        """
        Rebuild `poly` from the current (x, y, angle) values.

        Call this after mutating `x`, `y`, or `angle` in-place.
        """
        self.poly = make_tree_polygon(self.x, self.y, self.angle)


# ---------------------------------------------------------------------------
# Convenience types / exports
# ---------------------------------------------------------------------------

Point2D = Tuple[float, float]

__all__ = [
    "TREE_TEMPLATE_VERTS",
    "TREE_RADIUS",
    "make_tree_polygon_local",
    "make_tree_polygon",
    "TreePose",
    "Point2D",
]
