# Food Encounter Analysis

Analysis pipeline for C. elegans food encounter experiments. Takes raw tracked CSVs through masking, merging, curve fitting, permutation testing, and visualization.

## Pipeline Overview

```
Raw tracking CSVs
  → automatic_mask_borders.py   (create food mask, flag centroid/nose on food)
  → merge_files.py              (wide → long format, compute food_encounter)
  → bootstrap_food.py           (Richards curve fitting + bootstrap)
  → permutation_food.py         (permutation testing on bootstrap results)
  → graph_viewer_food.py        (visualize fits, traces, violins)
```

---

## Scripts

### `automatic_mask_borders.py`
Tkinter GUI for drawing or auto-generating a food mask from a video frame image. Processes tracking CSVs against the mask to produce per-worm, per-frame flags:
- `worm_k_centroid_on_food` — centroid vs. mask
- `worm_k_food_encounter` (nose vs. mask) — `nose_on_food` in the output

Optionally uses a SAM (Segment Anything) checkpoint from `./segmentation/` for automatic mask generation. Works without SAM — you can load a mask PNG manually.

**Input:** tracking CSV (wide format) + video frame image
**Output:** wide-format CSV with centroid and nose flags appended per worm

---

### `merge_files.py`
Tkinter GUI that converts wide-format tracked CSVs (one column group per worm) into a single long-format composite CSV. Supports manual metadata entry or automatic metadata inference from standardized filenames (`PC1_5.28.2025_m_wt_3hr`).

Key output column: `food_encounter` — labeled `'food'` only at the frame where the nose first crosses onto food while the centroid is still off food (true biological entry event).

Also supports appending new assays to an existing composite.

**Input:** directory of wide-format CSVs (output of tracking pipeline)
**Output:** `composite.csv` (long format, one row per worm per frame)

---

### `bootstrap_food.py`
Tkinter GUI for fitting Richards curves (4-parameter logistic) to C. elegans speed data aligned to food encounter. Uses bootstrap bagging (default 5000 replicates) to estimate parameter distributions.

Produces a permutation package containing:
- `bootstrap_results_combined.csv`
- `on_food_windowed.csv`
- `off_food_individual_baselines.csv`
- `metadata.json`

**Input:** `composite.csv` from `merge_files.py`
**Output:** bootstrap results directory + permutation package

---

### `bootstrap_food_vectorized.py`
Vectorized version of `bootstrap_food.py`. Faster on large datasets via numpy-based curve fitting. Same inputs, outputs, and permutation package format.

---

### `permutation_food.py`
Tkinter GUI for post-hoc permutation testing on bootstrap results. Provides both Standard and Max-T permutation tests for comparing groups (genotype, sex, treatment).

**Input:** permutation package from `bootstrap_food.py`
**Output:** permutation test results and plots

---

### `graph_viewer_food.py`
Tkinter + Matplotlib GUI (Richards Viewer v5) for visualizing bootstrap analysis results. Features include:
- Mean ± 95% CI trace plots
- Richards curve overlays
- Violin plots with bootstrap CI error bars
- Per-group color customization
- SVG export (transparent background, compatible with PowerPoint/LibreOffice)

**Input:** bootstrap results CSVs from `bootstrap_food.py`

---

## Installation

```
pip install -r requirements.txt
```

> **Linux users:** Tkinter is not pip-installable. Install via `sudo apt-get install python3-tk`.
>
> **SAM / GPU:** `automatic_mask_borders.py` can use Segment Anything for auto-masking. This requires a SAM checkpoint (`.pth`) placed in a `./segmentation/` folder next to the script. The app runs without it. The default torch install is CPU-only — see [PyTorch installation](https://pytorch.org/get-started/locally/) for GPU builds.
