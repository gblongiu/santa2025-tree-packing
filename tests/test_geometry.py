"""
Tests for santa2025.geometry

These tests focus on:
- Basic shape of the canonical tree polygon
- Correct computation of TREE_RADIUS
- Rotation / translation behavior of make_tree_polygon
- TreePose convenience helpers
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from santa2025.geometry import (
    TREE_TEMPLATE_VERTS,
    TREE_RADIUS,
    make_tree_polygon_local,
    make_tree_polygon,
    TreePose,
)


def test_tree_template_verts_shape_and_type():
    assert isinstance(TREE_TEMPLATE_VERTS, np.ndarray)
    # We hard-coded 15 vertices in geometry.py
    assert TREE_TEMPLATE_VERTS.shape[0] == 15
    assert TREE_TEMPLATE_VERTS.shape[1] == 2


def test_make_tree_polygon_local_is_polygon_and_valid():
    poly = make_tree_polygon_local()
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    # Polygon should have at least 3 distinct points
    assert len(poly.exterior.coords) >= 4  # closed ring repeats first point


def test_tree_radius_matches_max_vertex_norm():
    norms = np.linalg.norm(TREE_TEMPLATE_VERTS, axis=1)
    expected = float(norms.max())
    assert pytest.approx(TREE_RADIUS, rel=1e-9, abs=1e-12) == expected


def test_make_tree_polygon_translation_only():
    # Zero rotation: polygon should be a pure translation of local shape
    x, y, angle = 1.5, -0.7, 0.0
    local = make_tree_polygon_local()
    world = make_tree_polygon(x, y, angle)

    lx, ly = np.array(local.exterior.xy)
    wx, wy = np.array(world.exterior.xy)

    # Subtract translation and compare coordinates modulo closing point
    # Note: last point repeats first, so we can ignore that difference by length.
    assert len(lx) == len(wx)
    assert len(ly) == len(wy)

    lx = lx[:-1]
    ly = ly[:-1]
    wx = wx[:-1]
    wy = wy[:-1]

    # Undo translation
    wx_shifted = wx - x
    wy_shifted = wy - y

    assert np.allclose(lx, wx_shifted, atol=1e-9)
    assert np.allclose(ly, wy_shifted, atol=1e-9)


def test_make_tree_polygon_rotation_preserves_distances():
    # Rotation around origin should preserve radius of each vertex from (x, y)
    x, y = 0.3, -0.4
    angle = 37.0
    world = make_tree_polygon(x, y, angle)

    wx, wy = np.array(world.exterior.xy)
    # Ignore last repeated point
    wx = wx[:-1]
    wy = wy[:-1]

    # Distances from (x, y) to each vertex
    dists = np.sqrt((wx - x) ** 2 + (wy - y) ** 2)

    # Distances in local coords are distances from (0, 0)
    norms = np.linalg.norm(TREE_TEMPLATE_VERTS, axis=1)

    assert np.allclose(dists, norms, atol=1e-9)


def test_treepose_from_params_and_update():
    x0, y0, angle0 = 0.0, 0.0, 0.0
    pose = TreePose.from_params(x0, y0, angle0)

    # Basic checks
    assert isinstance(pose.poly, Polygon)
    assert pose.x == x0
    assert pose.y == y0
    assert pose.angle == angle0

    # Move pose and update
    pose.x = 1.0
    pose.y = -0.5
    pose.angle = 45.0
    pose.update()

    # Distances from new center should still match template norms
    wx, wy = np.array(pose.poly.exterior.xy)
    wx = wx[:-1]
    wy = wy[:-1]
    dists = np.sqrt((wx - pose.x) ** 2 + (wy - pose.y) ** 2)
    norms = np.linalg.norm(TREE_TEMPLATE_VERTS, axis=1)

    assert np.allclose(dists, norms, atol=1e-9)
