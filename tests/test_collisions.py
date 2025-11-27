"""
Tests for non-overlap (collision) detection utilities.

These tests focus on:
- has_any_collision correctly detecting overlaps
- Treating "touching" polygons (shared edge / point) as non-colliding
- Basic sanity checks with TreePose-based layouts
"""

from __future__ import annotations

import pytest
from shapely.geometry import box

from santa2025.packers.local_search import has_any_collision
from santa2025.geometry import TreePose, make_tree_polygon


def _make_box_pose(xmin, ymin, xmax, ymax) -> TreePose:
    """
    Helper: build a TreePose whose polygon is a simple rectangle (box).

    We ignore the angle and use x/y as 0 here since the collision logic
    only looks at the `poly` attribute.
    """
    poly = box(xmin, ymin, xmax, ymax)
    return TreePose(x=0.0, y=0.0, angle=0.0, poly=poly)


def test_no_collision_for_single_tree():
    poses = [_make_box_pose(0.0, 0.0, 1.0, 1.0)]
    assert has_any_collision(poses) is False


def test_non_overlapping_boxes_have_no_collision():
    poses = [
        _make_box_pose(0.0, 0.0, 1.0, 1.0),
        _make_box_pose(2.0, 2.0, 3.0, 3.0),
    ]
    assert has_any_collision(poses) is False


def test_overlapping_boxes_are_detected_as_collision():
    # These boxes share an overlapping area (not just a boundary)
    poses = [
        _make_box_pose(0.0, 0.0, 1.0, 1.0),
        _make_box_pose(0.5, 0.5, 1.5, 1.5),
    ]
    assert has_any_collision(poses) is True


def test_touching_boxes_are_not_considered_collision():
    # Second box touches the first exactly at x=1 edge
    poses_edge_touch = [
        _make_box_pose(0.0, 0.0, 1.0, 1.0),
        _make_box_pose(1.0, 0.0, 2.0, 1.0),
    ]
    assert has_any_collision(poses_edge_touch) is False

    # Second box touches the first at a single corner point (1, 1)
    poses_point_touch = [
        _make_box_pose(0.0, 0.0, 1.0, 1.0),
        _make_box_pose(1.0, 1.0, 2.0, 2.0),
    ]
    assert has_any_collision(poses_point_touch) is False


def test_real_tree_polygons_overlap_and_non_overlap():
    # Two trees at the same position and angle must collide
    pose_a = TreePose.from_params(0.0, 0.0, 0.0)
    pose_b = TreePose.from_params(0.0, 0.0, 0.0)
    assert has_any_collision([pose_a, pose_b]) is True

    # Two trees far apart should be non-colliding
    pose_c = TreePose.from_params(0.0, 0.0, 0.0)
    pose_d = TreePose.from_params(10.0, 10.0, 0.0)
    assert has_any_collision([pose_c, pose_d]) is False
