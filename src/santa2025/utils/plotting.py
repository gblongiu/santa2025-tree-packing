"""
Visualization helpers for the Santa 2025 Christmas Tree Packing project.

These helpers are thin convenience wrappers around matplotlib for:
- Plotting a single layout of trees with an optional bounding square.
- Plotting multiple layouts side-by-side for comparison.
- Inspecting the canonical tree shape in local coordinates.

All functions are designed to work with the `TreePose` objects defined in
`santa2025.geometry` and with standard matplotlib Axes objects.

Typical usage in a notebook
---------------------------

    import matplotlib.pyplot as plt
    from santa2025.geometry import TreePose
    from santa2025.utils.plotting import plot_layout

    poses = ...  # List[TreePose]
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_layout(poses, ax=ax, title="My layout")

You remain in control of figure creation and display.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import Polygon

from ..geometry import TreePose, make_tree_polygon_local


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_union_bounds_from_poses(poses: Sequence[TreePose]) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box (minx, miny, maxx, maxy) of the union of all
    polygons in `poses`.
    """
    union = unary_union([p.poly for p in poses])
    return union.bounds  # type: ignore[return-value]


def _plot_polygon(ax, poly: Polygon, **kwargs) -> None:
    """
    Plot a single shapely Polygon on the given Axes.
    """
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, alpha=kwargs.pop("alpha", 0.4))
    ax.plot(xs, ys, linewidth=kwargs.pop("linewidth", 0.8))


# ---------------------------------------------------------------------------
# Public plotting helpers
# ---------------------------------------------------------------------------

def plot_tree_local_shape(ax=None, show_origin: bool = True, title: str = "Tree shape (local coords)"):
    """
    Plot the canonical tree polygon in local coordinates.

    Parameters
    ----------
    ax:
        Optional matplotlib Axes. If None, a new figure and axes are created.
    show_origin:
        If True, draw crosshairs at the origin (0, 0).
    title:
        Plot title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    poly = make_tree_polygon_local()
    _plot_polygon(ax, poly)

    minx, miny, maxx, maxy = poly.bounds
    pad_x = 0.2 * (maxx - minx)
    pad_y = 0.2 * (maxy - miny)

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="box")
    if show_origin:
        ax.axhline(0.0, linewidth=0.5)
        ax.axvline(0.0, linewidth=0.5)
    ax.set_title(title)

    return ax


def plot_layout(
    poses: Sequence[TreePose],
    ax=None,
    title: Optional[str] = None,
    show_square: bool = True,
    padding: float = 0.5,
):
    """
    Plot a single layout of trees, optionally with the bounding square.

    Parameters
    ----------
    poses:
        Sequence of TreePose objects to plot.
    ax:
        Optional matplotlib Axes. If None, a new figure and axes are created.
    title:
        Optional plot title.
    show_square:
        If True, draw the smallest axis-aligned square that contains all trees.
    padding:
        Extra margin to add around the bounding square.
    """
    if not poses:
        raise ValueError("plot_layout called with an empty list of poses.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Plot each tree polygon
    for p in poses:
        _plot_polygon(ax, p.poly)

    # Compute union bounds
    minx, miny, maxx, maxy = _get_union_bounds_from_poses(poses)
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)

    if show_square:
        ax.add_patch(
            plt.Rectangle(
                (minx, miny),
                side,
                side,
                fill=False,
                linestyle="--",
                linewidth=2,
            )
        )

    ax.set_xlim(minx - padding, minx + side + padding)
    ax.set_ylim(miny - padding, miny + side + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    return ax


def plot_layouts_grid(
    layouts: Sequence[Sequence[TreePose]],
    titles: Optional[Sequence[str]] = None,
    ncols: int = 2,
    figsize_per_plot: Tuple[float, float] = (5.0, 5.0),
    show_square: bool = True,
    padding: float = 0.5,
):
    """
    Plot multiple layouts in a grid of subplots for visual comparison.

    Parameters
    ----------
    layouts:
        Sequence of layouts, each a sequence of TreePose objects.
    titles:
        Optional sequence of titles, one per layout.
    ncols:
        Number of columns in the subplot grid.
    figsize_per_plot:
        Size of each subplot in inches (width, height).
    show_square:
        Whether to draw the bounding square in each subplot.
    padding:
        Extra margin around the bounding square in each subplot.

    Returns
    -------
    (fig, axes):
        The matplotlib Figure and array of Axes.
    """
    num = len(layouts)
    if num == 0:
        raise ValueError("plot_layouts_grid called with an empty list of layouts.")

    if titles is not None and len(titles) != num:
        raise ValueError("If provided, 'titles' must match the number of layouts.")

    nrows = (num + ncols - 1) // ncols
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    if isinstance(axes, plt.Axes):
        axes = [[axes]]  # single subplot corner case

    axes_flat = [ax for row in axes for ax in (row if isinstance(row, (list, tuple)) else [row])]

    for i, poses in enumerate(layouts):
        ax = axes_flat[i]
        title_i = titles[i] if titles is not None else None
        plot_layout(poses, ax=ax, title=title_i, show_square=show_square, padding=padding)

    # Hide any unused axes
    for j in range(num, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig, axes
    

__all__ = [
    "plot_tree_local_shape",
    "plot_layout",
    "plot_layouts_grid",
]
