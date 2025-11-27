# Santa 2025 – Christmas Tree Packing (Local Project)

This repository contains my local PyCharm project for the **Kaggle “Santa 2025 – Christmas Tree Packing Challenge”**. The goal is to arrange 1–200 identical Christmas tree toys as tightly as possible inside an axis-aligned square, and then submit the tree centers and rotations in the required CSV format.

The focus here is:

- A **unique codebase**, not derived from the starter notebook.
- A **hexagonal-lattice baseline** for non-overlapping initial layouts.
- **Local search refinement** (hill climbing / simulated annealing style) to shrink each bounding square.
- Clean structure for experiments, plotting, and scoring.

This project is meant to be hackable: easy to plug in new packing strategies and compare them with the current best.

---

## Competition Summary

In the competition, each puzzle consists of **n trees** (for n from 1 to 200). For each tree, you must output:

- `x`, `y` – 2D coordinates of the **center of the top of the trunk**  
- `deg` – rotation angle in degrees  
- All numeric values are written as strings prefixed by `s`, e.g. `s0.123456`

The evaluation:

- For each n, the metric computes the **smallest axis-aligned square** that contains all n trees.
- Let `s_n` be that square’s side length, and `n` be the number of trees.
- The contribution of each puzzle is `s_n² / n`.
- The final score is the **sum over n = 1…200**. Lower is better.

This repo implements our own packing algorithms + a local clone of the metric so we can iterate quickly without uploading to Kaggle every time.

---

## Repository Layout

```text
santa2025-tree-packing/
  README.md
  requirements.txt
  pyproject.toml          # optional; may be used for poetry / modern packaging
  .gitignore

  data/
    raw/
      sample_submission.csv        # Kaggle sample
      metric_notebook.ipynb        # official metric or starter notebook (for reference only)
      # train/ or other files if the competition adds them later
    submissions/
      .gitkeep                     # placeholder so Git tracks the folder

  notebooks/
    00_metric_smoke_test.ipynb     # verify local metric matches Kaggle
    01_explore_tree_geometry.ipynb # visualize the tree shape, bounding circle, etc
    02_hex_lattice_baseline.ipynb  # experiments for hex packing
    03_local_search_experiments.ipynb

  src/
    santa2025/                     # mark "src" as Sources Root in PyCharm
      __init__.py
      config.py                    # seeds, global constants, paths
      geometry.py                  # tree polygon, circumscribed radius, helper functions
      evaluation.py                # local metric implementation (score computation)
      submission.py                # high-level glue to build all layouts and write a CSV

      packers/
        __init__.py
        hex_lattice.py             # main “unique” baseline: hex lattice + orientation pattern
        local_search.py            # hill climbing / annealing style refinement
        radial_baseline.py         # optional: alternative radial/greedy packer for comparison
        patterns.py                # experimental tilings, lattices, or parameterized patterns

      utils/
        __init__.py
        io.py                      # I/O utilities (load sample, save submissions, path helpers)
        plotting.py                # visualization helpers (matplotlib)
        timing.py                  # simple timing/profiling tools

  scripts/
    make_submission.py             # CLI: run best packer + refinement, save submission CSV
    score_submission.py            # CLI: load CSV, evaluate score with local metric

  tests/
    __init__.py
    test_geometry.py               # tests for tree polygon, radius, rotations, etc.
    test_collisions.py             # tests for non-overlap detection
    test_metric.py                 # confirm local metric aligns with Kaggle metric on samples
