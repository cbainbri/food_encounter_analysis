#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Richards Viewer v5 (Tk + Matplotlib)

Changes from v4:
  - Trace statistics (mean, 95% CI) pre-computed and cached at load time.
    _plot_traces / _plot_off_traces read from cache — no groupby on every
    refresh, making redraws near-instant.
  - Groups panel, colour swatches, curve overlay toggle, violin options, and
    OFF baseline source moved into a separate "Configure Groups & Display"
    Toplevel dialog opened by one button.  Left panel is now just:
    Load / file status / y-axis / display / figure size / export.
  - Bootstrap CI ribbon: 2.5/97.5 percentile across replicate means (correct).
  - Overlay: filters unconverged rows and handles bool-vs-string converged col.
  - Per-tab Y-axis limits (stored/restored on tab switch).
  - Violin error bars use bootstrap 95% CI (percentiles, not std/sqrt(N)).
  - Gaussian ribbon smoothing option.
  - SVG export: transparent background + ungroupable in PowerPoint/LibreOffice.
  - Re-entrant refresh guard prevents tick/font interactions.
  - Font sizes auto-scale with figure height (configurable).
  - Per-tab tick spacing and Y decimal format (blank = auto).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────────────
# GraphPad Prism–style rcParams
# ─────────────────────────────────────────────────────────────────────────────
_PRISM_RC = {
    "figure.facecolor":     "none",
    "axes.facecolor":       "white",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       1.5,
    "axes.labelsize":       12,
    "axes.labelweight":     "normal",
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.width":    1.5,
    "ytick.major.width":    1.5,
    "xtick.minor.width":    1.0,
    "ytick.minor.width":    1.0,
    "xtick.major.size":     5,
    "ytick.major.size":     5,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "lines.linewidth":      2.0,
    "lines.solid_capstyle": "round",
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "legend.frameon":       False,
    "legend.fontsize":      9,
    "axes.grid":            False,
    "pdf.fonttype":         42,
    "svg.fonttype":         "none",
    "savefig.dpi":          300,
    "savefig.facecolor":    "none",
    "figure.dpi":           100,
}
plt.rcParams.update(_PRISM_RC)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FIXED_SEX_ORDER       = ["h", "m"]
FIXED_GENOTYPE_ORDER  = ["wt", "tph-1"]
FIXED_TREATMENT_ORDER = ["fed", "30min", "3hr"]

DEFAULT_PALETTE = [
    # Hermaphrodites: fed, 30min, 3hr
    "#DB7373", "#DD4848", "#A00000",
    # Males: fed, 30min, 3hr
    "#2166AC", "#0000FF", "#000099",
    # Extras
    "#33A02C", "#8B3FA8", "#FF7F00", "#1B9EBF", "#E69F00", "#CC79A7",
]

TRACE_X_MIN = -30
TRACE_X_MAX  =  30
SEM_ALPHA    = 0.28

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def group_label(sex, genotype, treatment):
    return f"{sex.upper()} : {genotype} : {treatment}"

def safe_float(x, default=np.nan):
    try:    return float(x)
    except: return default

def logistic_curve(t, y_initial, y_final, B, M):
    A = y_final - y_initial
    with np.errstate(over="ignore", invalid="ignore"):
        denom = np.clip(1.0 + np.exp(-B * (t - M)), 1e-10, 1e10)
        raw   = y_initial + A / denom
    return np.where(np.isfinite(raw), raw, np.nan)

def mean_bootstrap_curve(t, group_params):
    """Vectorized: evaluate logistic for all bootstrap rows at once."""
    yi = group_params["y_initial"].values[:, None]
    yf = group_params["y_final"].values[:, None]
    B  = group_params["B"].values[:, None]
    M  = group_params["M"].values[:, None]
    A  = yf - yi
    with np.errstate(over="ignore", invalid="ignore"):
        denom  = np.clip(1.0 + np.exp(-B * (t[None, :] - M)), 1e-10, 1e10)
        curves = yi + A / denom
    curves = np.where(np.isfinite(curves), curves, np.nan)
    return np.nanmean(curves, axis=0)


def build_overlay_cache(fit_params: pd.DataFrame, t_fine: np.ndarray) -> dict:
    """Pre-compute mean + CI/SEM bands of bootstrap curves per group.
    Called once at load time.
    Each cache entry: (mean_curve, lo_ci, hi_ci, lo_sem, hi_sem)
      CI  = 2.5/97.5 percentile across replicate curves
      SEM = mean ± std(curves) / sqrt(N)
    """
    cache = {}
    if fit_params is None:
        return cache
    needed = {"y_initial", "y_final", "B", "M"}
    if not needed.issubset(fit_params.columns):
        return cache
    fp = fit_params.copy()
    if "converged" in fp.columns:
        fp = fp[fp["converged"].astype(str).str.strip().str.lower() == "true"]
    fp = fp.dropna(subset=["y_initial", "y_final", "B", "M"])
    for (sex, genotype, treatment), sub in fp.groupby(["sex", "strain_genotype", "treatment"]):
        if sub.empty:
            continue
        # Evaluate every bootstrap replicate's curve  (n_reps × n_timepoints)
        yi = sub["y_initial"].values[:, None]
        yf = sub["y_final"].values[:, None]
        B  = sub["B"].values[:, None]
        M  = sub["M"].values[:, None]
        A  = yf - yi
        with np.errstate(over="ignore", invalid="ignore"):
            denom  = np.clip(1.0 + np.exp(-B * (t_fine[None, :] - M)), 1e-10, 1e10)
            curves = yi + A / denom
        curves   = np.where(np.isfinite(curves), curves, np.nan)
        mean_c   = np.nanmean(curves, axis=0)
        if not np.any(np.isfinite(mean_c)):
            continue
        lo_ci    = np.nanpercentile(curves, 2.5,  axis=0)
        hi_ci    = np.nanpercentile(curves, 97.5, axis=0)
        n        = np.sum(np.isfinite(curves), axis=0)
        std_c    = np.nanstd(curves, axis=0, ddof=1)
        sem_c    = np.where(n > 1, std_c / np.sqrt(n), np.nan)
        lo_sem   = mean_c - sem_c
        hi_sem   = mean_c + sem_c
        cache[(str(sex), str(genotype), str(treatment))] = (
            mean_c, lo_ci, hi_ci, lo_sem, hi_sem
        )
    return cache


def compute_trace_stats(df: pd.DataFrame,
                        smooth_sigma: float = 0.0) -> Dict[Tuple, Tuple]:
    """
    Pre-compute (t, mu, lo, hi) for every (sex, genotype, treatment) group.
    For individual traces: mean +/- SEM across animals.
    For bootstrap traces: mean + 2.5/97.5 percentile CI across replicate means.
    smooth_sigma > 0 applies Gaussian smoothing to mu/lo/hi.
    """
    cache = {}
    has_animal   = {"assay_num", "track_num"}.issubset(df.columns)
    is_bootstrap = "bootstrap_replicate" in df.columns

    work = df.copy()
    work["time_rel"] = pd.to_numeric(work["time_rel"], errors="coerce")
    work["speed"]    = pd.to_numeric(work["speed"],    errors="coerce")
    work = work.dropna(subset=["time_rel", "speed"])
    work = work[(work["time_rel"] >= TRACE_X_MIN) & (work["time_rel"] <= TRACE_X_MAX)]

    for (sex, genotype, treatment), sub in work.groupby(["sex", "strain_genotype", "treatment"]):
        if has_animal:
            sub = sub.copy()
            sub["_aid"] = sub["assay_num"].astype(str) + "|" + sub["track_num"].astype(str)
            per_a = sub.groupby(["_aid", "time_rel"])["speed"].mean().reset_index()
            agg   = per_a.groupby("time_rel")["speed"].agg(["mean", "sem"]).sort_index()
            t  = agg.index.values
            mu = agg["mean"].values
            lo = mu - np.nan_to_num(agg["sem"].values)
            hi = mu + np.nan_to_num(agg["sem"].values)
        elif is_bootstrap:
            per_rep = sub.groupby(["bootstrap_replicate", "time_rel"])["speed"].mean().reset_index()
            grp = per_rep.groupby("time_rel")["speed"]
            t  = grp.mean().index.values
            mu = grp.mean().values
            lo = grp.quantile(0.025).values
            hi = grp.quantile(0.975).values
        else:
            agg = sub.groupby("time_rel")["speed"].agg(["mean", "sem"]).sort_index()
            t  = agg.index.values
            mu = agg["mean"].values
            lo = mu - np.nan_to_num(agg["sem"].values)
            hi = mu + np.nan_to_num(agg["sem"].values)

        if smooth_sigma > 0:
            mu = gaussian_filter1d(mu, sigma=smooth_sigma)
            lo = gaussian_filter1d(lo, sigma=smooth_sigma)
            hi = gaussian_filter1d(hi, sigma=smooth_sigma)

        cache[(str(sex), str(genotype), str(treatment))] = (t, mu, lo, hi)
    return cache


def _postprocess_svg(svg_bytes: bytes) -> bytes:
    """
    Post-process matplotlib SVG for PowerPoint / LibreOffice import:
      1. Remove patch_1/patch_2 background rects -> truly transparent.
      2. Clear residual white fills.
      3. Unwrap <g id="figure_1"> so elements are ungroupable.
    """
    import xml.etree.ElementTree as ET
    import io as _io

    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("",      SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ET.register_namespace("rdf",   "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    ET.register_namespace("dc",    "http://purl.org/dc/elements/1.1/")
    ET.register_namespace("cc",    "http://creativecommons.org/ns#")

    root = ET.fromstring(svg_bytes)

    def _rm_bg(parent):
        for child in list(parent):
            if child.get("id", "") in ("patch_1", "patch_2"):
                parent.remove(child)
            else:
                _rm_bg(child)
    _rm_bg(root)

    for el in root.iter():
        s = el.get("style", "")
        if s:
            for white in ("fill: #ffffff", "fill:#ffffff", "fill: white", "fill:white"):
                s = s.replace(white, "fill: none")
            el.set("style", s)

    for child in list(root):
        if child.get("id") == "figure_1":
            idx = list(root).index(child)
            root.remove(child)
            for i, grandchild in enumerate(child):
                root.insert(idx + i, grandchild)
            break

    buf = _io.BytesIO()
    ET.ElementTree(root).write(buf, xml_declaration=True, encoding="utf-8")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Simple Tk-native color picker (avoids KDE/system dialog)
# ─────────────────────────────────────────────────────────────────────────────
_PICKER_SWATCHES = [
    # Row 1 — Hermaphrodite reds (fed=#DB7373, 30min=#DD4848, 3hr=#A00000) + shades
    "#DB7373","#DD4848","#A00000","#E8303A","#C0392B","#922B21","#7B241C","#641E16",
    # Row 2 — Male blues (fed=#2166AC, 30min=#0000FF, 3hr=#000099) + shades
    "#2166AC","#0000FF","#000099","#1B9EBF","#56B4E9","#003366","#4488CC","#001F6E",
    # Row 3 — reds / oranges / yellows
    "#FF0000","#CC0000","#990000","#FF6600","#FF9900","#FFCC00","#FFFF00","#CCCC00",
    # Row 2 — greens
    "#00FF00","#00CC00","#009900","#006600","#33A02C","#00FF99","#00CCAA","#009966",
    # Row 3 — blues / cyans
    "#0000FF","#0000CC","#000099","#2166AC","#1B9EBF","#00CCFF","#00FFFF","#0099CC",
    # Row 4 — purples / pinks
    "#FF00FF","#CC00CC","#990099","#8B3FA8","#CC79A7","#F0699B","#FF6699","#FF99CC",
    # Row 5 — browns / oranges
    "#663300","#996633","#B15928","#CC9966","#FF7F00","#FFAA00","#E69F00","#FFCC66",
    # Row 6 — greys / black / white
    "#000000","#333333","#666666","#999999","#BBBBBB","#CCCCCC","#DDDDDD","#FFFFFF",
    # Row 7 — palette defaults
    "#E8303A","#2166AC","#33A02C","#8B3FA8","#FF7F00","#1B9EBF","#F0699B","#B15928",
    "#56B4E9","#009E73","#E69F00","#CC79A7","#888888","#444444","#222222","#EEEEEE",
    # Sampled from your data traces
    "#DB7373","#DD4848","#A00000","#C0392B","#E74C3C","#922B21","#7B241C","#641E16",
]

def _ask_color(parent, initial_color: str = "#888888", title: str = "Choose colour") -> str | None:
    """
    Simple Tkinter-native color chooser.
    Returns the chosen hex string, or None if cancelled.
    """
    result = [None]

    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.resizable(False, False)
    dlg.grab_set()

    COLS = 8
    SW   = 28   # swatch size px

    # ── Swatch grid ──────────────────────────────────────────────────────
    ttk.Label(dlg, text="▶ Hermaphrodite reds: fed · 30 min · 3 hr",
              font=("", 8), foreground="#A00000").pack(anchor="w", padx=8, pady=(6, 0))
    ttk.Label(dlg, text="▶ Male blues: fed · 30 min · 3 hr",
              font=("", 8), foreground="#2166AC").pack(anchor="w", padx=8, pady=(0, 2))
    grid = ttk.Frame(dlg, padding=8)
    grid.pack()

    current_hex = [initial_color.upper()]

    preview_var = tk.StringVar(value=current_hex[0])

    def _pick(hex_col):
        current_hex[0] = hex_col.upper()
        preview_var.set(current_hex[0])
        hex_e.delete(0, tk.END)
        hex_e.insert(0, current_hex[0])
        preview_lbl.config(bg=current_hex[0])

    for idx, col in enumerate(_PICKER_SWATCHES):
        r, c = divmod(idx, COLS)
        btn = tk.Button(grid, bg=col, width=2, relief="flat", bd=1,
                        activebackground=col,
                        command=lambda h=col: _pick(h))
        btn.grid(row=r, column=c, padx=1, pady=1, ipadx=2, ipady=4)

    # ── Hex entry + preview ───────────────────────────────────────────────
    bot = ttk.Frame(dlg, padding=(8, 0, 8, 8))
    bot.pack(fill=tk.X)

    ttk.Label(bot, text="Hex:").pack(side=tk.LEFT)
    hex_e = ttk.Entry(bot, width=9)
    hex_e.insert(0, current_hex[0])
    hex_e.pack(side=tk.LEFT, padx=(2, 6))

    preview_lbl = tk.Label(bot, bg=initial_color, width=4, relief="sunken", bd=2)
    preview_lbl.pack(side=tk.LEFT, padx=(0, 6))

    def _hex_changed(*_):
        val = hex_e.get().strip()
        if not val.startswith("#"):
            val = "#" + val
        try:
            dlg.winfo_rgb(val)   # raises if invalid
            current_hex[0] = val.upper()
            preview_lbl.config(bg=current_hex[0])
        except Exception:
            pass

    hex_e.bind("<KeyRelease>", _hex_changed)

    def _ok():
        result[0] = current_hex[0]
        dlg.destroy()

    def _cancel():
        dlg.destroy()

    ttk.Button(bot, text="OK",     command=_ok).pack(side=tk.RIGHT, padx=(4, 0))
    ttk.Button(bot, text="Cancel", command=_cancel).pack(side=tk.RIGHT)

    dlg.wait_window()
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GroupConfig:
    sex:       str
    genotype:  str
    treatment: str
    visible:   bool = True
    color:     str  = "#888888"
    # Per-group display toggles
    show_raw_mean: bool = True
    show_raw_sem:  bool = True
    show_fit_line: bool = False
    show_fit_ci:   bool = False

    @property
    def key(self): return (self.sex, self.genotype, self.treatment)

    @property
    def label(self): return group_label(self.sex, self.genotype, self.treatment)


@dataclass
class LoadedData:
    original_traces:   Optional[pd.DataFrame] = None
    bootstrap_traces:  Optional[pd.DataFrame] = None
    fit_params:        Optional[pd.DataFrame] = None
    off_baseline_boot: Optional[pd.DataFrame] = None
    off_baseline_ind:  Optional[pd.DataFrame] = None
    off_traces:        Optional[pd.DataFrame] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
class RichardsViewerApp:
    def __init__(self, root: tk.Tk):
        # Guard MUST be first — refresh() is called at end of __init__
        self._in_refresh = False

        self.root = root
        self.root.title("Richards Viewer v5")

        self.data   = LoadedData()
        self.groups: Dict[Tuple, GroupConfig] = {}

        # Pre-computed stat caches — rebuilt only on load
        self._on_cache:      Dict = {}
        self._off_cache:     Dict = {}
        self._overlay_t      = np.linspace(-30, 30, 400)
        self._overlay_cache: Dict = {}

        # ── Settings (shared with Configure dialog) ───────────────────────
        self.show_overlay     = tk.BooleanVar(value=True)
        self.show_raw_mean    = tk.BooleanVar(value=True)
        self.show_raw_sem     = tk.BooleanVar(value=True)
        self.show_overlay_ci  = tk.BooleanVar(value=False)
        self.overlay_ci_mode  = tk.StringVar(value="ci")   # "ci" or "sem"
        self.param_var      = tk.StringVar(value="y_initial")
        self.violin_auto_w  = tk.BooleanVar(value=True)
        self.violin_w       = tk.StringVar(value="0.5")
        self.violin_auto_bw = tk.BooleanVar(value=False)
        self.violin_bw      = tk.StringVar(value="0.3")
        self.min_spacing    = tk.StringVar(value="0.8")
        self.off_source     = tk.StringVar(value="bootstrap")
        self.smooth_sigma   = tk.StringVar(value="0")

        # ── Y-axis: widgets show/edit the ACTIVE tab's state ──────────────
        # _tab_ylim_state[i] = (auto:bool, ymin:str, ymax:str)
        self._tab_ylim_state = [(True, "0", "0.35")] * 4
        self._active_tab_idx = 0

        self.auto_ylim   = tk.BooleanVar(value=True)
        self.show_xlabel = tk.BooleanVar(value=True)
        self.show_legend = tk.BooleanVar(value=True)

        # ── Per-tab title overrides ───────────────────────────────────────
        self.tab_titles = [tk.StringVar() for _ in range(4)]

        # ── Font sizes ────────────────────────────────────────────────────
        self.font_auto     = tk.BooleanVar(value=True)
        self.font_title_sz = tk.StringVar(value="13")
        self.font_label_sz = tk.StringVar(value="12")
        self.font_tick_sz  = tk.StringVar(value="10")

        # ── Per-tab tick spacing (blank or 0 = matplotlib auto) ───────────
        self.ytick_step = [tk.StringVar(value="") for _ in range(4)]
        self.xtick_step = [tk.StringVar(value="") for _ in range(4)]
        # Y decimal places: blank = ScalarFormatter auto, digit = FormatStrFormatter
        self.ydecimal_n = [tk.StringVar(value="") for _ in range(4)]

        self._cfg_win = None
        self._cfg_display_vars: Dict = {}   # key -> {field: BooleanVar}

        self._build_ui()
        self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main, width=240)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)
        self._build_left(left)
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_tabs(right)

    def _build_left(self, p):
        ttk.Button(p, text="Load CSVs  (Auto-Detect)",
                   command=self.load_csvs).pack(fill=tk.X, pady=(0, 4))

        sf = ttk.LabelFrame(p, text="Loaded files", padding=4)
        sf.pack(fill=tk.X, pady=(0, 6))
        self.lbl_orig = ttk.Label(sf, text="Orig. traces:      \u2717", foreground="gray")
        self.lbl_boot = ttk.Label(sf, text="Bootstrap traces:  \u2717", foreground="gray")
        self.lbl_fit  = ttk.Label(sf, text="Fit params:        \u2717", foreground="gray")
        self.lbl_offb = ttk.Label(sf, text="OFF baseline boot: \u2717", foreground="gray")
        self.lbl_offi = ttk.Label(sf, text="OFF baseline ind:  \u2717", foreground="gray")
        self.lbl_offt = ttk.Label(sf, text="OFF traces (opt):  \u2717", foreground="gray")
        for lbl in (self.lbl_orig, self.lbl_boot, self.lbl_fit,
                    self.lbl_offb, self.lbl_offi, self.lbl_offt):
            lbl.pack(anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Button(p, text="\u2699  Configure Groups & Display",
                   command=self._open_configure).pack(fill=tk.X, pady=(0, 6))
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=6)

        # Y-axis (active tab) — widgets are backed by _tab_ylim_state
        yf = ttk.LabelFrame(p, text="Y-axis  (active tab)", padding=4)
        yf.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(yf, text="Auto", variable=self.auto_ylim,
                        command=self._on_ylim_change).pack(anchor="w")
        row = ttk.Frame(yf); row.pack(fill=tk.X)
        ttk.Label(row, text="Min").pack(side=tk.LEFT)
        self.ymin_e = ttk.Entry(row, width=7)
        self.ymin_e.insert(0, "0"); self.ymin_e.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="Max").pack(side=tk.LEFT)
        self.ymax_e = ttk.Entry(row, width=7)
        self.ymax_e.insert(0, "0.35"); self.ymax_e.pack(side=tk.LEFT, padx=2)
        for w in (self.ymin_e, self.ymax_e):
            w.bind("<KeyRelease>", lambda _: self._on_ylim_change())
            w.bind("<FocusOut>",   lambda _: self._on_ylim_change())

        df_ = ttk.LabelFrame(p, text="Display", padding=4)
        df_.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(df_, text="X-axis labels", variable=self.show_xlabel,
                        command=self.refresh).pack(anchor="w")
        ttk.Checkbutton(df_, text="Legend",        variable=self.show_legend,
                        command=self.refresh).pack(anchor="w")

        ff = ttk.LabelFrame(p, text="Figure size  (active tab, inches)", padding=4)
        ff.pack(fill=tk.X, pady=(0, 4))
        self.fig_tab_lbl = ttk.Label(ff, text="", font=("", 8), foreground="gray")
        self.fig_tab_lbl.pack(anchor="w")
        row = ttk.Frame(ff); row.pack(fill=tk.X)
        ttk.Label(row, text="W").pack(side=tk.LEFT)
        self.fig_w = ttk.Entry(row, width=6); self.fig_w.insert(0, "10")
        self.fig_w.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="H").pack(side=tk.LEFT)
        self.fig_h = ttk.Entry(row, width=6); self.fig_h.insert(0, "6")
        self.fig_h.pack(side=tk.LEFT, padx=2)
        ttk.Label(ff, text="Tip: same W and H \u2192 square axes area",
                  font=("", 7), foreground="gray").pack(anchor="w")
        ttk.Button(ff, text="Apply to this tab",
                   command=self._apply_fig_size).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(ff, text="Apply to all tabs",
                   command=self._apply_fig_size_all).pack(fill=tk.X, pady=(2, 0))

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Button(p, text="Export figure  (PNG / SVG / PDF)",
                   command=self._export_figure).pack(fill=tk.X, pady=(0, 10))

    def _build_tabs(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_traces  = ttk.Frame(self.nb)
        self.tab_fit     = ttk.Frame(self.nb)
        self.tab_offbase = ttk.Frame(self.nb)
        self.tab_offtr   = ttk.Frame(self.nb)

        self.nb.add(self.tab_traces,  text="Encounter Traces")
        self.nb.add(self.tab_fit,     text="Fit Parameters  (violin)")
        self.nb.add(self.tab_offbase, text="OFF Baseline  (violin)")
        self.nb.add(self.tab_offtr,   text="OFF Traces")
        self.nb.bind("<<NotebookTabChanged>>", lambda _: self._on_tab_change())

        # left=0.15 gives adequate room for y-axis labels even at small sizes
        VIOLIN_MARGINS = dict(left=0.15, right=0.97, top=0.93, bottom=0.32)
        TRACE_MARGINS  = dict(left=0.15, right=0.97, top=0.92, bottom=0.12)
        WORKSPACE_BG   = "#ffffff"

        self._fig_margins   = {}
        self._fig_canvas    = {}
        self._fig_workspace = {}
        self._fig_win_id    = {}

        def _make_tab(tab, margins, default_w=10, default_h=6):
            workspace = tk.Canvas(tab, bg=WORKSPACE_BG, highlightthickness=0)
            workspace.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

            fig = Figure(figsize=(default_w, default_h), facecolor="none")
            ax  = fig.add_subplot(111)
            ax.set_facecolor("none")
            fig.subplots_adjust(**margins)

            mpl_canvas = FigureCanvasTkAgg(fig, master=workspace)
            widget     = mpl_canvas.get_tk_widget()
            dpi        = fig.dpi
            px_w, px_h = int(default_w * dpi), int(default_h * dpi)
            widget.config(width=px_w, height=px_h, bg=WORKSPACE_BG)
            win_id = workspace.create_window(0, 0, window=widget, anchor="nw")

            def _center(event, ws=workspace, wid=win_id, fw=px_w, fh=px_h):
                x = max(0, (event.width  - fw) // 2)
                y = max(0, (event.height - fh) // 2)
                ws.coords(wid, x, y)
            workspace.bind("<Configure>", _center)

            toolbar_frame = ttk.Frame(tab)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            NavigationToolbar2Tk(mpl_canvas, toolbar_frame)

            self._fig_margins[fig]   = margins
            self._fig_canvas[fig]    = mpl_canvas
            self._fig_workspace[fig] = workspace
            self._fig_win_id[fig]    = win_id
            return fig, ax, mpl_canvas

        self.fig_tr,  self.ax_tr,  self.canvas_tr  = _make_tab(self.tab_traces,  TRACE_MARGINS)
        self.fig_fit, self.ax_fit, self.canvas_fit  = _make_tab(self.tab_fit,     VIOLIN_MARGINS)
        self.fig_ob,  self.ax_ob,  self.canvas_ob   = _make_tab(self.tab_offbase, VIOLIN_MARGINS)
        self.fig_ot,  self.ax_ot,  self.canvas_ot   = _make_tab(self.tab_offtr,   TRACE_MARGINS)

    # ─────────────────────────────────────────────────────────────────────────
    # Configure Groups & Display dialog
    # ─────────────────────────────────────────────────────────────────────────
    def _open_configure(self):
        if self._cfg_win is not None and self._cfg_win.winfo_exists():
            self._cfg_win.lift(); self._cfg_win.focus_force(); return

        win = tk.Toplevel(self.root)
        win.title("Configure Groups & Display")
        win.resizable(True, True)
        win.geometry("420x720")
        self._cfg_win = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_configure(win))

        outer = ttk.Frame(win); outer.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner  = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        # Mouse wheel — named functions so they survive the Destroy event
        def _wheel(e):
            try:
                if not canvas.winfo_exists(): return
                if   e.num == 4: canvas.yview_scroll(-1, "units")
                elif e.num == 5: canvas.yview_scroll( 1, "units")
                else:            canvas.yview_scroll(int(-1 * e.delta / 120), "units")
            except Exception:
                pass

        def _bind_wheel(_e=None):
            if canvas.winfo_exists():
                canvas.bind_all("<MouseWheel>", _wheel)
                canvas.bind_all("<Button-4>",   _wheel)
                canvas.bind_all("<Button-5>",   _wheel)

        def _unbind_wheel(_e=None):
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except Exception:
                pass

        canvas.bind("<Enter>",  _bind_wheel)
        canvas.bind("<Leave>",  _unbind_wheel)
        win.bind("<Destroy>",   lambda _: _unbind_wheel())

        p = inner

        # ── Groups ────────────────────────────────────────────────────────
        hdr = ttk.Frame(p); hdr.pack(fill=tk.X, padx=8, pady=(8, 2))
        ttk.Label(hdr, text="Groups", font=("", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(hdr, text="None", width=5,
                   command=lambda: self._select_all_groups(False)).pack(side=tk.RIGHT)
        ttk.Button(hdr, text="All",  width=4,
                   command=lambda: self._select_all_groups(True)).pack(side=tk.RIGHT, padx=(0, 2))
        ttk.Label(p, text="  \u2611 = visible   \u25a0 = colour",
                  font=("", 8), foreground="gray").pack(anchor="w", padx=8)
        self._cfg_group_frame = ttk.Frame(p)
        self._cfg_group_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
        self._rebuild_cfg_groups()
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Set all groups (bulk apply) ───────────────────────────────────
        of = ttk.LabelFrame(p, text="Set all groups:", padding=6)
        of.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(of, text="Applies selected display options to every group at once.",
                  font=("", 7), foreground="gray", wraplength=340).pack(anchor="w", pady=(0, 4))

        def _apply_all():
            rm = self.show_raw_mean.get()
            rs = self.show_raw_sem.get()
            fl = self.show_overlay.get()
            fc = self.show_overlay_ci.get()
            for gc in self.groups.values():
                gc.show_raw_mean = rm
                gc.show_raw_sem  = rs and rm
                gc.show_fit_line = fl
                gc.show_fit_ci   = fc and fl
            # sync BooleanVars to reflect forced-off children
            self.show_raw_sem.set(rs and rm)
            self.show_overlay_ci.set(fc and fl)
            self._rebuild_cfg_groups()
            self._update_trace_toggle_states()
            self.refresh()

        row_g = ttk.Frame(of); row_g.pack(fill=tk.X)
        ttk.Checkbutton(row_g, text="trace",
                        variable=self.show_raw_mean).pack(side=tk.LEFT)
        ttk.Checkbutton(row_g, text="SEM",
                        variable=self.show_raw_sem).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Separator(row_g, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Checkbutton(row_g, text="model",
                        variable=self.show_overlay).pack(side=tk.LEFT)
        ttk.Checkbutton(row_g, text="fit±",
                        variable=self.show_overlay_ci).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(of, text="Apply to all groups", command=_apply_all).pack(
            fill=tk.X, pady=(6, 0))

        # ── Model fit CI/SEM mode (global) ────────────────────────────────
        mf = ttk.LabelFrame(p, text="Model fit ribbon mode  (global)", padding=6)
        mf.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Radiobutton(mf, text="95% CI  (2.5–97.5 pct of fit curves)",
                        variable=self.overlay_ci_mode, value="ci",
                        command=self.refresh).pack(anchor="w")
        ttk.Radiobutton(mf, text="SEM  (std / √N of fit curves)",
                        variable=self.overlay_ci_mode, value="sem",
                        command=self.refresh).pack(anchor="w")
        ttk.Label(mf, text="(applies to all groups with fit± enabled)",
                  font=("", 7), foreground="gray").pack(anchor="w")
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Violin options ────────────────────────────────────────────────
        vf = ttk.LabelFrame(p, text="Violin options", padding=6)
        vf.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Label(vf, text="Parameter  (Fit Parameters tab):").pack(anchor="w")
        param_cb = ttk.Combobox(vf, textvariable=self.param_var, state="readonly",
                                values=["y_initial", "y_final", "A", "B", "M",
                                        "r_squared", "food_detection_score",
                                        "food_encounter_score"])
        param_cb.pack(fill=tk.X, pady=(0, 6))
        param_cb.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        ttk.Label(vf, text="Violin width:").pack(anchor="w")
        ttk.Checkbutton(vf, text="Auto", variable=self.violin_auto_w,
                        command=self.refresh).pack(anchor="w")
        row = ttk.Frame(vf); row.pack(fill=tk.X)
        ttk.Label(row, text="Scale (0.1\u20131.0)").pack(side=tk.LEFT)
        ve = ttk.Entry(row, textvariable=self.violin_w, width=6)
        ve.pack(side=tk.LEFT, padx=2)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            ve.bind(ev, lambda _: self.refresh())

        ttk.Label(vf, text="KDE bandwidth:", font=("", 8)).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(vf, text="Auto (Scott's rule)", variable=self.violin_auto_bw,
                        command=self.refresh).pack(anchor="w")
        row = ttk.Frame(vf); row.pack(fill=tk.X)
        ttk.Label(row, text="BW").pack(side=tk.LEFT)
        bwe = ttk.Entry(row, textvariable=self.violin_bw, width=6)
        bwe.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(0.2 tight \u2013 0.6 smooth)").pack(side=tk.LEFT)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            bwe.bind(ev, lambda _: self.refresh())

        ttk.Label(vf, text="Min spacing (in/violin):", font=("", 8)).pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(vf); row.pack(fill=tk.X)
        mse = ttk.Entry(row, textvariable=self.min_spacing, width=6)
        mse.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(figure expands only if crowded)").pack(side=tk.LEFT)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            mse.bind(ev, lambda _: self.refresh())
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── OFF baseline source ───────────────────────────────────────────
        ob = ttk.LabelFrame(p, text="OFF baseline source", padding=6)
        ob.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Radiobutton(ob, text="Bootstrap",  variable=self.off_source,
                        value="bootstrap",  command=self.refresh).pack(anchor="w")
        ttk.Radiobutton(ob, text="Individual", variable=self.off_source,
                        value="individual", command=self.refresh).pack(anchor="w")
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Trace ribbon smoothing ────────────────────────────────────────
        sf = ttk.LabelFrame(p, text="Trace ribbon smoothing", padding=6)
        sf.pack(fill=tk.X, padx=8, pady=(0, 6))
        row = ttk.Frame(sf); row.pack(fill=tk.X)
        ttk.Label(row, text="Gaussian \u03c3:", font=("", 9)).pack(side=tk.LEFT)
        sm_e = ttk.Entry(row, textvariable=self.smooth_sigma, width=5)
        sm_e.pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="(0 = off, 1\u20133 = subtle, 5+ = heavy)",
                  font=("", 7), foreground="gray").pack(side=tk.LEFT)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            sm_e.bind(ev, lambda _: self._on_smooth_change())
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Tick Spacing ──────────────────────────────────────────────────
        # Blank or 0 = matplotlib auto.  Positive number = use that step/decimals.
        tf = ttk.LabelFrame(p, text="Tick Spacing  (per tab)", padding=6)
        tf.pack(fill=tk.X, padx=8, pady=(0, 6))
        for col, lbl in enumerate(["Tab", "Y step", "X step", "Y decimals"]):
            ttk.Label(tf, text=lbl, font=("", 8, "bold"),
                      width=16 if col == 0 else 8).grid(
                row=0, column=col, sticky="w", padx=(0 if col == 0 else 4, 0))
        ttk.Label(tf, text="(blank = auto)", font=("", 7),
                  foreground="gray").grid(row=0, column=4, sticky="w", padx=4)

        for i, tab_name in enumerate(self._TAB_NAMES):
            r = i + 1
            ttk.Label(tf, text=tab_name, font=("", 8), foreground="gray",
                      width=16).grid(row=r, column=0, sticky="w", pady=2)
            ys_e = ttk.Entry(tf, textvariable=self.ytick_step[i], width=7)
            ys_e.grid(row=r, column=1, padx=4)
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                ys_e.bind(ev, lambda _: self.refresh())
            xs_e = ttk.Entry(tf, textvariable=self.xtick_step[i], width=7)
            xs_e.grid(row=r, column=2, padx=4)
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                xs_e.bind(ev, lambda _: self.refresh())
            yn_e = ttk.Entry(tf, textvariable=self.ydecimal_n[i], width=4)
            yn_e.grid(row=r, column=3, padx=4)
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                yn_e.bind(ev, lambda _: self.refresh())
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Labels & Fonts ────────────────────────────────────────────────
        lf = ttk.LabelFrame(p, text="Labels & Fonts", padding=6)
        lf.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(lf, text="Plot titles  (blank = auto):", font=("", 8, "bold")).pack(anchor="w")
        tab_default_titles = ["Encounter-centred speed", "Fit parameters",
                              "OFF-food baseline speed", "OFF-food traces"]
        for i, (var, _dflt) in enumerate(zip(self.tab_titles, tab_default_titles)):
            row = ttk.Frame(lf); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{self._TAB_NAMES[i]}:", width=18,
                      font=("", 8), foreground="gray").pack(side=tk.LEFT)
            e = ttk.Entry(row, textvariable=var)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                e.bind(ev, lambda _: self.refresh())

        ttk.Separator(lf, orient="horizontal").pack(fill=tk.X, pady=(6, 4))
        ttk.Label(lf, text="Font sizes:", font=("", 8, "bold")).pack(anchor="w")
        ttk.Checkbutton(lf, text="Auto-scale with figure height",
                        variable=self.font_auto,
                        command=self.refresh).pack(anchor="w")
        ttk.Label(lf, text="(title \u2248 h\u00d72.1 \u2022 axis label \u2248 h\u00d71.8 \u2022 tick \u2248 h\u00d71.5)",
                  font=("", 7), foreground="gray").pack(anchor="w")
        fsz = ttk.Frame(lf); fsz.pack(fill=tk.X, pady=(4, 0))
        for col, (lbl, var) in enumerate([("Title", self.font_title_sz),
                                           ("Axis lbl", self.font_label_sz),
                                           ("Tick", self.font_tick_sz)]):
            ttk.Label(fsz, text=lbl, font=("", 8)).grid(row=0, column=col*2, padx=(4, 0), sticky="w")
            e = ttk.Entry(fsz, textvariable=var, width=4)
            e.grid(row=0, column=col*2+1, padx=(2, 4))
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                e.bind(ev, lambda _: self.refresh())

        ttk.Button(p, text="Close",
                   command=lambda: self._close_configure(win)).pack(pady=(0, 10))

    def _close_configure(self, win):
        win.destroy()
        self._cfg_win = None

    def _update_trace_toggle_states(self):
        """No-op: per-group dependency is handled inline in _rebuild_cfg_groups.
        Kept so any existing calls don't break."""
        pass

    def _rebuild_cfg_groups(self):
        if not hasattr(self, "_cfg_group_frame"):
            return
        for w in self._cfg_group_frame.winfo_children():
            w.destroy()
        self._cfg_group_vars    = {}
        self._cfg_group_btns    = {}
        self._cfg_display_vars  = {}
        for sex in FIXED_SEX_ORDER:
            for genotype in FIXED_GENOTYPE_ORDER:
                for trt in FIXED_TREATMENT_ORDER:
                    key = (sex, genotype, trt)
                    if key not in self.groups:
                        continue
                    gc  = self.groups[key]
                    var = tk.BooleanVar(value=gc.visible)
                    self._cfg_group_vars[key] = var

                    # ── top row: visibility + colour ──────────────────────
                    row = ttk.Frame(self._cfg_group_frame)
                    row.pack(fill=tk.X, pady=(2, 0))

                    def _toggle(k=key, v=var):
                        self.groups[k].visible = v.get()
                        self.refresh()

                    ttk.Checkbutton(row, text=gc.label, variable=var,
                                    command=_toggle).pack(side=tk.LEFT, fill=tk.X, expand=True)
                    btn = tk.Button(row, bg=gc.color, width=3, relief="raised",
                                    command=lambda k=key: self._pick_color(k))
                    btn.pack(side=tk.RIGHT, padx=(2, 0))
                    self._cfg_group_btns[key] = btn

                    # ── sub-row: per-group display toggles ────────────────
                    sub = ttk.Frame(self._cfg_group_frame)
                    sub.pack(fill=tk.X, padx=(18, 0), pady=(0, 2))

                    dv = {
                        "show_raw_mean": tk.BooleanVar(value=gc.show_raw_mean),
                        "show_raw_sem":  tk.BooleanVar(value=gc.show_raw_sem),
                        "show_fit_line": tk.BooleanVar(value=gc.show_fit_line),
                        "show_fit_ci":   tk.BooleanVar(value=gc.show_fit_ci),
                    }
                    self._cfg_display_vars[key] = dv

                    # Forward-declare sem_cb and ci_cb so commands can ref them
                    sem_cb_holder = [None]
                    ci_cb_holder  = [None]

                    def _on_raw_mean(k=key, v=dv):
                        val = v["show_raw_mean"].get()
                        self.groups[k].show_raw_mean = val
                        if not val:
                            v["show_raw_sem"].set(False)
                            self.groups[k].show_raw_sem = False
                        if sem_cb_holder[0]:
                            sem_cb_holder[0].config(
                                state="normal" if val else "disabled")
                        self.refresh()

                    def _on_raw_sem(k=key, v=dv):
                        self.groups[k].show_raw_sem = v["show_raw_sem"].get()
                        self.refresh()

                    def _on_fit_line(k=key, v=dv):
                        val = v["show_fit_line"].get()
                        self.groups[k].show_fit_line = val
                        if not val:
                            v["show_fit_ci"].set(False)
                            self.groups[k].show_fit_ci = False
                        if ci_cb_holder[0]:
                            ci_cb_holder[0].config(
                                state="normal" if val else "disabled")
                        self.refresh()

                    def _on_fit_ci(k=key, v=dv):
                        self.groups[k].show_fit_ci = v["show_fit_ci"].get()
                        self.refresh()

                    ttk.Checkbutton(sub, text="trace",
                                    variable=dv["show_raw_mean"],
                                    command=_on_raw_mean).pack(side=tk.LEFT)

                    sem_cb = ttk.Checkbutton(sub, text="SEM",
                                             variable=dv["show_raw_sem"],
                                             command=_on_raw_sem,
                                             state="normal" if gc.show_raw_mean else "disabled")
                    sem_cb.pack(side=tk.LEFT, padx=(4, 0))
                    sem_cb_holder[0] = sem_cb

                    ttk.Separator(sub, orient="vertical").pack(
                        side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

                    ttk.Checkbutton(sub, text="model",
                                    variable=dv["show_fit_line"],
                                    command=_on_fit_line).pack(side=tk.LEFT)

                    ci_cb = ttk.Checkbutton(sub, text="fit±",
                                            variable=dv["show_fit_ci"],
                                            command=_on_fit_ci,
                                            state="normal" if gc.show_fit_line else "disabled")
                    ci_cb.pack(side=tk.LEFT, padx=(4, 0))
                    ci_cb_holder[0] = ci_cb

                    ttk.Separator(self._cfg_group_frame,
                                  orient="horizontal").pack(fill=tk.X, pady=(2, 0))

    def _pick_color(self, key):
        gc     = self.groups[key]
        chosen = _ask_color(self.root, initial_color=gc.color,
                            title=f"Colour \u2014 {gc.label}")
        if chosen:
            gc.color = chosen
            if key in self._cfg_group_btns:
                self._cfg_group_btns[key].configure(bg=gc.color)
            self.refresh()

    def _select_all_groups(self, state: bool):
        for key, gc in self.groups.items():
            gc.visible = state
            if key in self._cfg_group_vars:
                self._cfg_group_vars[key].set(state)
        self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Group registry
    # ─────────────────────────────────────────────────────────────────────────
    def _populate_groups(self, df: pd.DataFrame):
        needed = {"sex", "strain_genotype", "treatment"}
        if not needed.issubset(df.columns):
            return
        changed = False
        for _, row in df[["sex", "strain_genotype", "treatment"]].drop_duplicates().iterrows():
            sex, gen, trt = str(row["sex"]), str(row["strain_genotype"]), str(row["treatment"])
            key = (sex, gen, trt)
            if key not in self.groups:
                idx = len(self.groups) % len(DEFAULT_PALETTE)
                self.groups[key] = GroupConfig(
                    sex=sex, genotype=gen, treatment=trt,
                    visible=True, color=DEFAULT_PALETTE[idx],
                    show_raw_mean=self.show_raw_mean.get(),
                    show_raw_sem =self.show_raw_sem.get(),
                    show_fit_line=self.show_overlay.get(),
                    show_fit_ci  =self.show_overlay_ci.get(),
                )
                changed = True
        if changed:
            self._rebuild_cfg_groups()

    def _visible_groups(self) -> List[GroupConfig]:
        out = []
        for sex in FIXED_SEX_ORDER:
            for genotype in FIXED_GENOTYPE_ORDER:
                for trt in FIXED_TREATMENT_ORDER:
                    key = (sex, genotype, trt)
                    if key in self.groups and self.groups[key].visible:
                        out.append(self.groups[key])
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # CSV classification & loading
    # ─────────────────────────────────────────────────────────────────────────
    def _classify(self, df, filepath):
        cols  = set(df.columns)
        fname = filepath.lower()
        if {"y_initial", "y_final", "B", "M", "r_squared", "bootstrap_replicate"}.issubset(cols):
            return "fit_params"
        if {"time_rel", "speed", "treatment", "sex", "strain_genotype"}.issubset(cols):
            if "off" in fname and any(k in fname for k in ("window", "trace", "windowed")):
                return "off_traces"
            if {"assay_num", "track_num"}.issubset(cols):
                return "original_traces"
            if "bootstrap_replicate" in cols:
                return "bootstrap_traces"
            if "original" in fname:
                return "original_traces"
            return "bootstrap_traces"
        if {"baseline_speed", "treatment", "sex", "strain_genotype"}.issubset(cols):
            if "bootstrap_replicate" in cols or "sample_size" in cols:
                return "off_baseline_boot"
            if {"assay_num", "track_num"}.issubset(cols):
                return "off_baseline_ind"
            return "off_baseline_boot" if "bootstrap" in fname else "off_baseline_ind"
        return None

    def load_csvs(self):
        files = filedialog.askopenfilenames(
            title="Select Richards output CSVs",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not files:
            return
        loaded = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                messagebox.showerror("Load error", f"Could not read:\n{fp}\n\n{e}")
                continue
            kind = self._classify(df, fp)
            if kind is None:
                messagebox.showwarning("Unrecognized",
                    f"Could not classify CSV (unexpected columns):\n{fp}")
                continue
            if kind == "original_traces":
                self.data.original_traces = df; self._populate_groups(df)
                loaded.append("Original animal traces")
            elif kind == "bootstrap_traces":
                self.data.bootstrap_traces = df; self._populate_groups(df)
                loaded.append("Bootstrap traces")
            elif kind == "fit_params":
                self.data.fit_params = df; self._populate_groups(df)
                loaded.append("Fit parameters")
            elif kind == "off_baseline_boot":
                self.data.off_baseline_boot = df; self._populate_groups(df)
                loaded.append("OFF baseline \u2014 bootstrap")
            elif kind == "off_baseline_ind":
                self.data.off_baseline_ind = df; self._populate_groups(df)
                loaded.append("OFF baseline \u2014 individual")
            elif kind == "off_traces":
                self.data.off_traces = df; self._populate_groups(df)
                loaded.append("OFF traces")

        if loaded:
            self._rebuild_caches()

        self._update_status()
        self.refresh()
        # Show messagebox AFTER refresh so it doesn't block the first layout pass
        if loaded:
            messagebox.showinfo("Loaded",
                                "Loaded:\n" + "\n".join(f"  \u2022 {x}" for x in loaded))

    def _get_smooth_sigma(self) -> float:
        try:    return max(0.0, float(self.smooth_sigma.get()))
        except: return 0.0

    def _on_smooth_change(self):
        self._rebuild_caches()
        self.refresh()

    def _rebuild_caches(self):
        sigma = self._get_smooth_sigma()
        on_df = (self.data.original_traces
                 if self.data.original_traces is not None
                 else self.data.bootstrap_traces)
        self._on_cache      = compute_trace_stats(on_df, smooth_sigma=sigma) if on_df is not None else {}
        self._off_cache     = compute_trace_stats(self.data.off_traces, smooth_sigma=sigma) if self.data.off_traces is not None else {}
        self._overlay_cache = build_overlay_cache(self.data.fit_params, self._overlay_t)

    def _update_status(self):
        def _s(lbl, ok, yes, no):
            lbl.config(text=yes if ok else no, foreground="black" if ok else "gray")
        _s(self.lbl_orig,  self.data.original_traces   is not None, "Orig. traces:      \u2713", "Orig. traces:      \u2717")
        _s(self.lbl_boot,  self.data.bootstrap_traces  is not None, "Bootstrap traces:  \u2713", "Bootstrap traces:  \u2717")
        _s(self.lbl_fit,   self.data.fit_params        is not None, "Fit params:        \u2713", "Fit params:        \u2717")
        _s(self.lbl_offb,  self.data.off_baseline_boot is not None, "OFF baseline boot: \u2713", "OFF baseline boot: \u2717")
        _s(self.lbl_offi,  self.data.off_baseline_ind  is not None, "OFF baseline ind:  \u2713", "OFF baseline ind:  \u2717")
        _s(self.lbl_offt,  self.data.off_traces        is not None, "OFF traces (opt):  \u2713", "OFF traces (opt):  \u2717")

    # ─────────────────────────────────────────────────────────────────────────
    # Per-tab Y-axis state
    # ─────────────────────────────────────────────────────────────────────────
    def _save_ylim_state(self):
        i = self._active_tab_idx
        self._tab_ylim_state[i] = (
            self.auto_ylim.get(),
            self.ymin_e.get(),
            self.ymax_e.get(),
        )

    def _load_ylim_state(self, tab_idx: int):
        auto, ymin, ymax = self._tab_ylim_state[tab_idx]
        self.auto_ylim.set(auto)
        self.ymin_e.delete(0, tk.END); self.ymin_e.insert(0, ymin)
        self.ymax_e.delete(0, tk.END); self.ymax_e.insert(0, ymax)

    def _on_ylim_change(self):
        """User edited a Y-axis widget — save state then refresh."""
        self._save_ylim_state()
        self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Shared plot helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_ylim(self, ax):
        if self.auto_ylim.get():
            return
        lo = safe_float(self.ymin_e.get(), np.nan)
        hi = safe_float(self.ymax_e.get(), np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            ax.set_ylim(lo, hi)

    def _apply_ticks(self, ax, tab_idx: int):
        """Blank or 0 = matplotlib auto.  Positive number = use it."""
        import matplotlib.ticker as ticker
        import math

        # Y locator
        y_step = safe_float(self.ytick_step[tab_idx].get(), 0.0)
        if y_step > 0:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        else:
            ax.yaxis.set_major_locator(ticker.AutoLocator())

        # Y decimal formatter
        # If the user typed a decimal count, use it exactly.
        # If blank AND a y_step is set, infer the required decimal places from
        # the step so ticks like 0.05 never round to duplicate labels like 0.1, 0.1.
        # If blank AND no step, use ScalarFormatter (matplotlib auto).
        dp_str = self.ydecimal_n[tab_idx].get().strip()
        if dp_str:
            try:
                nd = max(0, int(dp_str))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{nd}f"))
            except (ValueError, TypeError):
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_useOffset(False)
        elif y_step > 0:
            # Auto-infer decimals from step: 0.05 → 2dp, 0.1 → 1dp, 0.25 → 2dp
            nd = max(0, math.ceil(-math.log10(y_step + 1e-12)))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{nd}f"))
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_useOffset(False)

        # X locator (trace tabs only — violin tabs manage integer x-ticks)
        if tab_idx in (0, 3):
            x_step = safe_float(self.xtick_step[tab_idx].get(), 0.0)
            if x_step > 0:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
            else:
                ax.xaxis.set_major_locator(ticker.AutoLocator())

    def _get_font_sizes(self, fig):
        """Return (title_sz, label_sz, tick_sz).
        Auto: proportional to figure height.  Manual: reads entry vars.
        _in_refresh is True here so the .set() calls don't recurse."""
        if self.font_auto.get():
            h  = fig.get_figheight()
            ts = max(10, round(h * 2.1))
            ls = max(9,  round(h * 1.8))
            ks = max(8,  round(h * 1.5))
            self.font_title_sz.set(str(ts))
            self.font_label_sz.set(str(ls))
            self.font_tick_sz.set(str(ks))
            return ts, ls, ks
        def _i(v, default):
            try:    return max(5, int(float(v.get())))
            except: return default
        return (_i(self.font_title_sz, 13),
                _i(self.font_label_sz, 12),
                _i(self.font_tick_sz,  10))

    def _get_tab_title(self, tab_idx, auto_title: str) -> str:
        override = self.tab_titles[tab_idx].get().strip()
        return override if override else auto_title

    def _apply_prism_spines(self, ax):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.5)
        ax.tick_params(axis="both", which="major", direction="out", width=1.5, length=5)
        ax.tick_params(axis="both", which="minor", direction="out", width=1.0, length=3)

    def _apply_display(self, ax, fig, has_legend=False):
        self._apply_prism_spines(ax)
        ts, ls, ks = self._get_font_sizes(fig)
        ax.tick_params(axis="both", labelsize=ks)
        ax.xaxis.label.set_fontsize(ls)
        ax.yaxis.label.set_fontsize(ls)
        ax.title.set_fontsize(ts)
        if not self.show_xlabel.get():
            ax.set_xticklabels([])
            ax.set_xlabel("")
        if has_legend:
            legend_fs = max(7, ks - 1)
            if self.show_legend.get():
                # str() cast: matplotlib 3.7+ returns Text object from get_label()
                labeled = [a for a in ax.get_children()
                           if hasattr(a, "get_label") and
                           not str(a.get_label()).startswith("_")]
                if labeled:
                    ax.legend(fontsize=legend_fs)
            else:
                leg = ax.get_legend()
                if leg: leg.remove()

    _TAB_NAMES = ["Encounter Traces", "Fit Parameters", "OFF Baseline", "OFF Traces"]

    def _resize_fig(self, fig, w, h):
        """Resize figure (inches) + widget (pixels) + recenter on workspace."""
        dpi = fig.dpi
        fig.set_size_inches(w, h)
        margins = self._fig_margins.get(fig, {})
        if margins:
            fig.subplots_adjust(**margins)
        px_w = int(w * dpi)
        px_h = int(h * dpi)
        canvas    = self._fig_canvas.get(fig)
        workspace = self._fig_workspace.get(fig)
        win_id    = self._fig_win_id.get(fig)
        if canvas is not None:
            widget = canvas.get_tk_widget()
            widget.config(width=px_w, height=px_h)
            if workspace is not None and win_id is not None:
                workspace.bind("<Configure>",
                    lambda e, wid=win_id, fw=px_w, fh=px_h, ws=workspace:
                        ws.coords(wid, max(0, (e.width-fw)//2), max(0, (e.height-fh)//2)))
                def _do_center(ws=workspace, wid=win_id, fw=px_w, fh=px_h):
                    ws_w = ws.winfo_width()
                    ws_h = ws.winfo_height()
                    if ws_w > 1 and ws_h > 1:
                        ws.coords(wid, max(0, (ws_w-fw)//2), max(0, (ws_h-fh)//2))
                    else:
                        ws.after(50, _do_center)
                _do_center()
            canvas.draw()

    def _active_fig(self):
        idx = self.nb.index(self.nb.select())
        return (self.fig_tr, self.fig_fit, self.fig_ob, self.fig_ot)[idx]

    def _on_tab_change(self):
        """Save current tab's Y-axis state, load new tab's, then refresh."""
        self._save_ylim_state()
        new_idx = self.nb.index(self.nb.select())
        self._active_tab_idx = new_idx
        self._load_ylim_state(new_idx)
        fig = self._active_fig()
        w, h = fig.get_size_inches()
        self.fig_w.delete(0, tk.END); self.fig_w.insert(0, f"{w:.1f}")
        self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h:.1f}")
        if hasattr(self, "fig_tab_lbl"):
            self.fig_tab_lbl.config(text=self._TAB_NAMES[new_idx])
        self.refresh()

    def _square_h(self, fig, w) -> float:
        """Compute H so axes AREA is square given W and this figure's margins."""
        m    = self._fig_margins.get(fig, {})
        ax_w = m.get("right", 0.97) - m.get("left",   0.15)
        ax_h = m.get("top",   0.93) - m.get("bottom", 0.12)
        return w * ax_w / ax_h if ax_h > 0 else w

    def _resolve_h(self, fig, w, h_entered) -> float:
        """If W and H are within 0.01" compensate margins for a square axes area."""
        if abs(w - h_entered) < 0.01:
            return self._square_h(fig, w)
        return h_entered

    def _apply_fig_size(self):
        w     = safe_float(self.fig_w.get(), 10.0)
        h_raw = safe_float(self.fig_h.get(),  6.0)
        h     = self._resolve_h(self._active_fig(), w, h_raw)
        if w > 0 and h > 0:
            self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h:.2f}")
            self._resize_fig(self._active_fig(), w, h)
            self.refresh()

    def _apply_fig_size_all(self):
        w     = safe_float(self.fig_w.get(), 10.0)
        h_raw = safe_float(self.fig_h.get(),  6.0)
        if w > 0:
            for fig in (self.fig_tr, self.fig_fit, self.fig_ob, self.fig_ot):
                h = self._resolve_h(fig, w, h_raw)
                self._resize_fig(fig, w, h)
            h_active = self._resolve_h(self._active_fig(), w, h_raw)
            self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h_active:.2f}")
            self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Violin helpers
    # ─────────────────────────────────────────────────────────────────────────
    _DEFAULT_MIN_SPACING = 0.8

    def _get_violin_width(self, n):
        if self.violin_auto_w.get():
            return 0.80
        return float(np.clip(safe_float(self.violin_w.get(), 0.5), 0.05, 1.5))

    def _get_violin_bw(self):
        if self.violin_auto_bw.get():
            return "scott"
        return float(np.clip(safe_float(self.violin_bw.get(), 0.3), 0.05, 2.0))

    def _auto_size_violin_fig(self, fig, n: int, vw: float):
        """Expand figure only if violins would be too crowded."""
        min_spacing  = float(np.clip(safe_float(self.min_spacing.get(),
                                                self._DEFAULT_MIN_SPACING), 0.1, 10.0))
        min_w_needed = (n * min_spacing) / 0.87
        if fig.get_figwidth() < min_w_needed:
            self._resize_fig(fig, min_w_needed, fig.get_figheight())

    def _draw_violin(self, ax, fig, data_list, labels, colors):
        n  = len(data_list)
        vw = self._get_violin_width(n)
        self._auto_size_violin_fig(fig, n, vw)

        parts = ax.violinplot(data_list, showmeans=False, showextrema=False,
                              widths=vw, bw_method=self._get_violin_bw())
        for i, col in enumerate(colors):
            parts["bodies"][i].set_facecolor(col)
            parts["bodies"][i].set_alpha(0.80)
            parts["bodies"][i].set_edgecolor("none")

        ax.set_xlim(0.5, n + 0.5)

        xs     = list(range(1, n + 1))
        means  = [np.mean(d) for d in data_list]
        # Bootstrap 95% CI via percentiles.
        # Do NOT divide by sqrt(N) — replicates are not independent.
        lo_err = [means[i] - np.percentile(d, 2.5)  for i, d in enumerate(data_list)]
        hi_err = [np.percentile(d, 97.5) - means[i] for i, d in enumerate(data_list)]
        ax.errorbar(xs, means, yerr=[lo_err, hi_err], fmt="o", color="black",
                    markersize=6, markerfacecolor="black", markeredgecolor="black",
                    linewidth=1.8, capsize=5, capthick=1.8, zorder=5,
                    label="mean \u00b1 95% CI (bootstrap)")
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────
    def _export_figure(self):
        idx  = self.nb.index(self.nb.select())
        figs = [self.fig_tr, self.fig_fit, self.fig_ob, self.fig_ot]
        fp   = filedialog.asksaveasfilename(
                   defaultextension=".png",
                   filetypes=[("PNG", "*.png"), ("SVG", "*.svg"),
                               ("PDF", "*.pdf"), ("All files", "*.*")])
        if not fp:
            return
        import os
        fig_to_save   = figs[idx]
        saved_margins = self._fig_margins.get(fig_to_save, {})
        # tight_layout before save so labels always fit regardless of figure size
        fig_to_save.tight_layout(pad=0.6)
        if os.path.splitext(fp)[1].lower() == ".svg":
            import io as _io
            buf = _io.BytesIO()
            fig_to_save.savefig(buf, format="svg", transparent=True)
            with open(fp, "wb") as _f:
                _f.write(_postprocess_svg(buf.getvalue()))
        else:
            fig_to_save.savefig(fp, dpi=150, transparent=True)
        # Restore fixed margins so interactive view is unchanged
        if saved_margins:
            fig_to_save.subplots_adjust(**saved_margins)
        messagebox.showinfo("Exported", f"Saved:\n{fp}")

    # ─────────────────────────────────────────────────────────────────────────
    # Refresh dispatcher
    # ─────────────────────────────────────────────────────────────────────────
    def refresh(self):
        if self._in_refresh:
            return
        self._in_refresh = True
        try:
            idx = self.nb.index(self.nb.select())
            if hasattr(self, "fig_tr"):
                fig = (self.fig_tr, self.fig_fit, self.fig_ob, self.fig_ot)[idx]
                w, h = fig.get_size_inches()
                self.fig_w.delete(0, tk.END); self.fig_w.insert(0, f"{w:.1f}")
                self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h:.1f}")
                if hasattr(self, "fig_tab_lbl"):
                    self.fig_tab_lbl.config(text=self._TAB_NAMES[idx])
            [self._plot_traces,
             self._plot_fit_violin,
             self._plot_off_baseline,
             self._plot_off_traces][idx]()
        finally:
            self._in_refresh = False

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 0 — Encounter traces  (unchanged from v5(1))
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_traces(self):
        ax = self.ax_tr
        ax.clear()

        if not self._on_cache:
            if self.data.original_traces is None and self.data.bootstrap_traces is None:
                ax.text(0.5, 0.5,
                        "Load  windowed_speed_data_original.csv\n"
                        "or windowed_speed_data.csv",
                        ha="center", va="center", fontsize=9, transform=ax.transAxes)
                self.canvas_tr.draw_idle()
                return
            self._rebuild_caches()

        using_orig = self.data.original_traces is not None
        ci_label   = "mean \u00b1 SEM" if using_orig else "mean \u00b1 95% CI (bootstrap)"

        ov_ci_mode = self.overlay_ci_mode.get()   # global: "ci" or "sem"

        plotted_gcs = []
        for gc in self._visible_groups():
            entry = self._on_cache.get(gc.key)
            if entry is None:
                continue
            t, mu, lo, hi = entry
            if gc.show_raw_sem and gc.show_raw_mean:
                ax.fill_between(t, lo, hi, color=gc.color, alpha=SEM_ALPHA,
                                linewidth=0.6, edgecolor=gc.color, antialiased=True)
            if gc.show_raw_mean:
                ax.plot(t, mu, color=gc.color, linewidth=2.0, label=gc.label)
            plotted_gcs.append(gc)

        if self._overlay_cache:
            for gc in plotted_gcs:
                if not gc.show_fit_line:
                    continue
                ov_entry = self._overlay_cache.get(gc.key)
                if ov_entry is None:
                    continue
                mean_c, lo_ci, hi_ci, lo_sem, hi_sem = ov_entry
                if gc.show_fit_ci:
                    lo_band = lo_ci if ov_ci_mode == "ci" else lo_sem
                    hi_band = hi_ci if ov_ci_mode == "ci" else hi_sem
                    ax.fill_between(self._overlay_t, lo_band, hi_band,
                                    color=gc.color, alpha=SEM_ALPHA * 0.55,
                                    linewidth=0, antialiased=True)
                ax.plot(self._overlay_t, mean_c, color=gc.color, linestyle="--",
                        linewidth=2.0, alpha=0.85, label=f"{gc.label}  (fit)")

        ax.axvline(0, color="#444444", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.set_xlim(TRACE_X_MIN, TRACE_X_MAX)
        ax.set_title(self._get_tab_title(0, f"Encounter-centred speed  ({ci_label})"))
        ax.set_xlabel("Time relative to food encounter (s)")
        ax.set_ylabel("Speed (mm/s)")
        self._apply_ylim(ax)
        self._apply_ticks(ax, 0)
        self._apply_display(ax, self.fig_tr, has_legend=True)
        self.canvas_tr.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 — Fit parameters violin  (unchanged from v5(1))
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_fit_violin(self):
        ax = self.ax_fit
        ax.clear()

        if self.data.fit_params is None:
            ax.text(0.5, 0.5, "Load  richards_fit_parameters.csv",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_fit.draw_idle()
            return

        df    = self.data.fit_params
        param = self.param_var.get()
        if param not in df.columns:
            ax.text(0.5, 0.5, f"Column '{param}' not found in fit parameters CSV",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_fit.draw_idle()
            return

        # Scores are ratios where lower = more slowing; invert so the plotted
        # quantity is "slowing increase" (0 = no slowing, 1 = fully stopped).
        INVERT_SCORES = {"food_detection_score", "food_encounter_score"}
        invert = param in INVERT_SCORES

        data_list, labels, colors = [], [], []
        for gc in self._visible_groups():
            vals = df[(df["sex"] == gc.sex) & (df["strain_genotype"] == gc.genotype) &
                      (df["treatment"] == gc.treatment)][param].dropna().values
            if len(vals) == 0:
                continue
            if invert:
                vals = 1.0 - vals
            data_list.append(vals); labels.append(gc.label); colors.append(gc.color)

        if not data_list:
            ax.text(0.5, 0.5, "No fit data for selected groups",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_fit.draw_idle()
            return

        self._draw_violin(ax, self.fig_fit, data_list, labels, colors)

        score_labels = {
            "food_detection_score": "1 \u2212 detection score  (slowing at detection)",
            "food_encounter_score": "1 \u2212 encounter score  (slowing at encounter)",
        }
        ylabel = score_labels.get(param, param)
        ax.set_title(self._get_tab_title(1, f"Fit parameter \u2014 {param}"))
        ax.set_ylabel(ylabel)
        self._apply_ylim(ax)
        self._apply_ticks(ax, 1)
        self._apply_display(ax, self.fig_fit, has_legend=False)
        self.canvas_fit.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 — OFF baseline violin  (unchanged from v5(1))
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_off_baseline(self):
        ax = self.ax_ob
        ax.clear()

        source = self.off_source.get()
        df     = (self.data.off_baseline_boot if source == "bootstrap"
                  else self.data.off_baseline_ind)
        if df is None:
            fname = ("off_food_bootstrap_baselines.csv" if source == "bootstrap"
                     else "off_food_individual_baselines.csv")
            ax.text(0.5, 0.5, f"Load  {fname}",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_ob.draw_idle()
            return

        if not {"baseline_speed", "sex", "treatment", "strain_genotype"}.issubset(df.columns):
            ax.text(0.5, 0.5, "OFF baseline CSV missing required columns",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_ob.draw_idle()
            return

        work = df.copy()
        work["baseline_speed"] = pd.to_numeric(work["baseline_speed"], errors="coerce")
        work = work.dropna(subset=["baseline_speed"])

        data_list, labels, colors = [], [], []
        for gc in self._visible_groups():
            vals = work[(work["sex"] == gc.sex) & (work["strain_genotype"] == gc.genotype) &
                        (work["treatment"] == gc.treatment)]["baseline_speed"].values
            if len(vals) == 0:
                continue
            data_list.append(vals); labels.append(gc.label); colors.append(gc.color)

        if not data_list:
            ax.text(0.5, 0.5, "No OFF baseline data for selected groups",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_ob.draw_idle()
            return

        self._draw_violin(ax, self.fig_ob, data_list, labels, colors)
        ax.set_title(self._get_tab_title(2, f"OFF-food baseline speed  ({source})"))
        ax.set_ylabel("baseline_speed  (mm/s)")
        self._apply_ylim(ax)
        self._apply_ticks(ax, 2)
        self._apply_display(ax, self.fig_ob, has_legend=False)
        self.canvas_ob.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — OFF food windowed traces  (unchanged from v5(1))
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_off_traces(self):
        ax = self.ax_ot
        ax.clear()

        if not self._off_cache:
            if self.data.off_traces is None:
                ax.text(0.5, 0.5,
                        "Load  off_food_windowed_speed_data.csv\n"
                        "(output by richards_speed_analysis_FINAL.py in bootstrap mode)\n\n"
                        "columns: time_rel, speed, bootstrap_replicate, sex, treatment, strain_genotype",
                        ha="center", va="center", fontsize=9, transform=ax.transAxes)
            else:
                self._rebuild_caches()
            self.canvas_ot.draw_idle()
            return

        for gc in self._visible_groups():
            entry = self._off_cache.get(gc.key)
            if entry is None:
                continue
            t, mu, lo, hi = entry
            if gc.show_raw_sem and gc.show_raw_mean:
                ax.fill_between(t, lo, hi, color=gc.color, alpha=SEM_ALPHA,
                                linewidth=0.6, edgecolor=gc.color, antialiased=True)
            if gc.show_raw_mean:
                ax.plot(t, mu, color=gc.color, linewidth=2.0, label=gc.label)

        ax.axvline(0, color="#444444", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.set_title(self._get_tab_title(3, "OFF-food traces  (mean \u00b1 95% CI)"))
        ax.set_xlabel("Time relative to encounter (s)")
        ax.set_ylabel("Speed (mm/s)")
        self._apply_ylim(ax)
        self._apply_ticks(ax, 3)
        self._apply_display(ax, self.fig_ot, has_legend=True)
        self.canvas_ot.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1280x820")
    RichardsViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
