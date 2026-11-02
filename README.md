# Food Encounter Analysis

Analysis pipeline for C. elegans food encounter experiments. Takes raw tracked CSVs through masking, merging, curve fitting, permutation testing, and visualization.

## Pipeline Overview

```
Raw tracking CSVs
  → automatic_mask_borders.py   (create food mask, flag centroid/nose on food)
  → tracking_pipeline/prep_track_data.py  (collect finished tracks as PC#_<folder>.csv)
  → tracking_pipeline/merge_files.py      (wide → long format, compute food_encounter)
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

### Building `composite.csv`
`merge_files.py` (and the `prep_track_data.py` step before it) now live in [tracking_pipeline](https://github.com/cbainbri/tracking_pipeline), shared with opto analysis. It computes the `food_encounter` column — labeled `'food'` only at the frame where the nose first crosses onto food while the centroid is still off food (true biological entry event).

---

### `bootstrap_food.py`
Tkinter GUI for fitting Richards curves (4-parameter logistic) to C. elegans speed data aligned to food encounter. Uses bootstrap bagging (default 5000 replicates) to estimate parameter distributions.

Produces a permutation package containing:
- `bootstrap_results_combined.csv`
- `on_food_windowed.csv`
- `off_food_individual_baselines.csv`
- `metadata.json`

**Input:** `composite.csv` from `tracking_pipeline/merge_files.py`
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

### Download the SAM model weights

`automatic_mask_borders.py` requires the Segment Anything Model (SAM) checkpoint to auto-generate food masks. The weights file is ~1.2 GB and is not stored in git — run this once after installing:

```
python download_models.py
```

This downloads `sam_vit_l_0b3195.pth` from Meta's servers into the `segmentation/` folder next to the script. `automatic_mask_borders.py` detects it automatically on launch — no configuration needed. If the file is already present it will not re-download.

> **Linux users:** Tkinter is not pip-installable. Install via `sudo apt-get install python3-tk`.
>
> **GPU:** The default torch install is CPU-only. See [PyTorch installation](https://pytorch.org/get-started/locally/) for GPU builds. GPU is not required — SAM runs on CPU, just slower.
