#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Richards & Double-Richards Curve Fitting - GUI Version [FIXED]
WITH PREPROCESSING (Outlier Filtering + Smoothing)

GUI application for fitting Richards curves to C. elegans speed data
with optional preprocessing steps.

BUGFIXES:
1. Bootstrap mode now correctly saves INDIVIDUAL animal OFF-food baselines
   in addition to bootstrap aggregates.
   
2. Permutation package now includes behavioral context scores 
   (food_detection_score, food_encounter_score) if OFF-food data is loaded.
   
Permutation package contains all required files:
  - bootstrap_results_combined.csv (ON + OFF food bootstrap results + behavioral scores)
  - on_food_windowed.csv (individual animal ON-food trajectories)
  - off_food_individual_baselines.csv (individual animal OFF-food baselines)
  - metadata.json

This ensures permutation testing can work with all parameters including derived
behavioral context scores that the Richards script calculates and visualizes.
"""

import tkinter as tk
import json
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import traceback
import warnings
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)
plt.rcParams["figure.dpi"] = 110


# =============================================================================
# Sorting and Ordering Functions
# =============================================================================

def identify_wild_type(genotypes):
    """Identify the wild-type genotype from a list of genotypes"""
    genotypes_clean = [str(g).strip().upper() for g in genotypes if pd.notna(g)]
    
    # Common WT identifiers (case-insensitive)
    wt_identifiers = ['N2', 'WT', 'WILD-TYPE', 'WILD TYPE', 'WILDTYPE']
    
    for wt_id in wt_identifiers:
        for orig_genotype in genotypes:
            if pd.notna(orig_genotype) and str(orig_genotype).strip().upper() == wt_id:
                return orig_genotype
    
    # If no match, return the first genotype (fallback)
    return genotypes[0] if len(genotypes) > 0 else None


def sort_genotypes(genotypes):
    """Sort genotypes with WT first, then alphabetically"""
    unique_genotypes = sorted(set(g for g in genotypes if pd.notna(g)))
    if len(unique_genotypes) == 0:
        return []
    
    wt = identify_wild_type(unique_genotypes)
    
    if wt is None:
        return sorted(unique_genotypes)
    
    # WT first, then others alphabetically
    other_genotypes = sorted([g for g in unique_genotypes if g != wt])
    return [wt] + other_genotypes


def sort_sexes(sexes):
    """Sort sexes with hermaphrodite first, then male, then others"""
    unique_sexes = sorted(set(s for s in sexes if pd.notna(s)))
    if len(unique_sexes) == 0:
        return []
    
    # Define preferred order
    sex_order = []
    
    # Hermaphrodite variants (case-insensitive matching)
    herm_variants = ['hermaphrodite', 'herm', 'h']
    for sex in unique_sexes:
        if str(sex).lower().strip() in herm_variants:
            sex_order.append(sex)
            break
    
    # Male variants
    male_variants = ['male', 'm']
    for sex in unique_sexes:
        if str(sex).lower().strip() in male_variants and sex not in sex_order:
            sex_order.append(sex)
            break
    
    # Add any remaining sexes alphabetically
    remaining = sorted([s for s in unique_sexes if s not in sex_order])
    return sex_order + remaining


def sort_treatments(treatments):
    """Sort treatments in biological order: fed, then increasing starvation"""
    unique_treatments = sorted(set(t for t in treatments if pd.notna(t)))
    if len(unique_treatments) == 0:
        return []
    
    # Define preferred order (fed first, then by time)
    treatment_order_map = {
        'fed': 0,
        'food': 0,
        '0min': 1,
        '0hr': 1,
        '30min': 2,
        '30m': 2,
        '1hr': 3,
        '1h': 3,
        '3hr': 4,
        '3h': 4,
        '6hr': 5,
        '6h': 5,
        '12hr': 6,
        '12h': 6,
        '24hr': 7,
        '24h': 7,
    }
    
    def get_sort_key(treatment):
        treatment_lower = str(treatment).lower().strip()
        if treatment_lower in treatment_order_map:
            return (treatment_order_map[treatment_lower], treatment)
        # Unknown treatments go at the end, sorted alphabetically
        return (1000, treatment)
    
    return sorted(unique_treatments, key=get_sort_key)


def create_hierarchical_groups(df, group_cols=['strain_genotype', 'sex', 'treatment']):
    """
    Create hierarchically ordered groups: Genotype → Sex → Treatment
    Returns list of tuples in the correct order
    """
    if df.empty:
        return []
    
    # Get unique values for each grouping level
    genotypes = sort_genotypes(df['strain_genotype'].unique())
    sexes = sort_sexes(df['sex'].unique())
    treatments = sort_treatments(df['treatment'].unique())
    
    # Build hierarchical order
    ordered_groups = []
    for genotype in genotypes:
        for sex in sexes:
            for treatment in treatments:
                # Check if this combination exists in the data
                mask = (
                    (df['strain_genotype'] == genotype) &
                    (df['sex'] == sex) &
                    (df['treatment'] == treatment)
                )
                if mask.any():
                    ordered_groups.append((treatment, sex, genotype))
    
    return ordered_groups


def format_group_label(treatment, sex, genotype):
    """Create a readable label for a group"""
    return f"{genotype}|{sex}|{treatment}"


# =============================================================================
# Preprocessing functions
# =============================================================================

def apply_speed_threshold_filter(
    df: pd.DataFrame,
    max_speed: float = 0.4,
    speed_col: str = "speed"
) -> pd.DataFrame:
    """
    Remove data points with speeds exceeding biological plausibility threshold.
    
    This is a hard cutoff to remove obviously erroneous tracking data before any analysis.
    Values are set to NaN rather than removing rows to preserve temporal structure.
    
    Args:
        df: DataFrame with speed column
        max_speed: Maximum plausible speed in mm/s (default 0.4)
        speed_col: Name of speed column
    
    Returns:
        DataFrame with extreme speeds set to NaN
    """
    df = df.copy()
    
    if speed_col not in df.columns:
        return df
    
    # Count points to be filtered
    n_total = df[speed_col].notna().sum()
    outlier_mask = df[speed_col] > max_speed
    n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        # Set outliers to NaN (preserves temporal structure)
        df.loc[outlier_mask, speed_col] = np.nan
        pct_filtered = (n_outliers / n_total * 100) if n_total > 0 else 0
        print(f"  Filtered {n_outliers} points exceeding {max_speed} mm/s ({pct_filtered:.2f}%)")
    
    return df


def filter_outlier_animals_by_parameter(
    df_results: pd.DataFrame,
    parameter: str = 'y_initial',
    percentile_lower: float = 1.25,
    percentile_upper: float = 98.75,
    mad_threshold: float = 2.5,
    group_cols: List[str] = ['treatment', 'sex', 'strain_genotype']
) -> pd.DataFrame:
    """
    Filter outlier animals based on fitted parameter values (e.g., y_initial).
    
    Uses hybrid percentile + MAD filtering within each group.
    Animals are filtered if flagged by BOTH methods (conservative).
    
    This filters ANIMALS (one row per animal) not FRAMES.
    
    Args:
        df_results: DataFrame with fitted parameters (one row per animal)
        parameter: Parameter to use for filtering (default 'y_initial')
        percentile_lower: Lower percentile threshold (default 1.25)
        percentile_upper: Upper percentile threshold (default 98.75)
        mad_threshold: MAD threshold (default 2.5)
        group_cols: Columns defining groups for filtering
    
    Returns:
        Filtered DataFrame (outlier animals removed)
    """
    df = df_results.copy()
    
    if parameter not in df.columns:
        print(f"  Warning: Parameter '{parameter}' not in results, skipping animal filtering")
        return df
    
    # Only filter converged fits (if 'converged' column exists)
    if 'converged' in df.columns:
        converged_mask = df['converged'] == True
        
        if converged_mask.sum() == 0:
            print(f"  No converged fits to filter")
            return df
        
        df_to_filter = df[converged_mask]
    else:
        # No converged column (e.g., OFF-food data which are just means, not fits)
        df_to_filter = df
    
    # Filter within each group
    filtered_dfs = []
    total_removed = 0
    
    for group_vals, group_df in df_to_filter.groupby(group_cols):
        values = group_df[parameter].values
        
        if len(values) < 3:
            # Too few animals to filter
            filtered_dfs.append(group_df)
            continue
        
        # Percentile-based outlier detection
        lower_pct = np.percentile(values, percentile_lower)
        upper_pct = np.percentile(values, percentile_upper)
        pct_outliers = (values < lower_pct) | (values > upper_pct)
        
        # MAD-based outlier detection
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        
        if mad > 0:
            lower_mad = median_val - mad_threshold * mad
            upper_mad = median_val + mad_threshold * mad
            mad_outliers = (values < lower_mad) | (values > upper_mad)
        else:
            mad_outliers = np.zeros(len(values), dtype=bool)
        
        # Keep only if flagged by BOTH methods (conservative)
        combined_outliers = pct_outliers & mad_outliers
        
        # Keep non-outliers
        keep_mask = ~combined_outliers
        filtered_group = group_df[keep_mask]
        filtered_dfs.append(filtered_group)
        
        n_removed = combined_outliers.sum()
        total_removed += n_removed
        
        if n_removed > 0:
            group_label = '|'.join(str(v) for v in group_vals)
            print(f"  {group_label}: Removed {n_removed}/{len(values)} outlier animals by {parameter}")
    
    # Add back non-converged fits if they exist (don't filter those)
    if 'converged' in df.columns:
        non_converged = df[~converged_mask]
        if len(non_converged) > 0:
            filtered_dfs.append(non_converged)
    
    result = pd.concat(filtered_dfs, ignore_index=True)
    
    if total_removed > 0:
        n_total = len(df_to_filter)
        print(f"  Total outlier animals removed: {total_removed}/{n_total}")
    
    return result


def apply_hybrid_outlier_filter(
    df: pd.DataFrame,
    value_col: str = "speed",
    group_cols: List[str] = ["assay_num", "track_num"],
    percentile_lower: float = 1.25,
    percentile_upper: float = 98.75,
    mad_threshold: float = 2.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hybrid outlier detection: percentile + MAD"""
    df = df.copy()
    df['is_outlier'] = False
    df['flagged_by'] = ''
    
    outlier_stats = []
    
    for group_id, group in df.groupby(group_cols):
        values = group[value_col].dropna()
        if len(values) < 3:
            continue
            
        lower_pct = np.percentile(values, percentile_lower)
        upper_pct = np.percentile(values, percentile_upper)
        pct_out = (values < lower_pct) | (values > upper_pct)
        
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        if mad > 0:
            lower_mad = median_val - mad_threshold * mad
            upper_mad = median_val + mad_threshold * mad
            mad_out = (values < lower_mad) | (values > upper_mad)
        else:
            mad_out = pd.Series(False, index=values.index)
        
        inter = pct_out & mad_out
        out_idx = values[inter].index
        df.loc[out_idx, 'is_outlier'] = True
        
        for idx in out_idx:
            methods = []
            if pct_out.loc[idx]:
                methods.append('percentile')
            if mad_out.loc[idx]:
                methods.append('MAD')
            df.loc[idx, 'flagged_by'] = '+'.join(methods)
        
        n_total = len(values)
        n_outliers = len(out_idx)
        pct_outliers = (n_outliers / n_total * 100) if n_total > 0 else 0
        
        outlier_stats.append({
            **dict(zip(group_cols, group_id if isinstance(group_id, tuple) else [group_id])),
            'n_points': n_total,
            'n_outliers': n_outliers,
            'pct_outliers': pct_outliers,
        })
    
    df_filtered = df[~df['is_outlier']].copy()
    df_filtered.drop(columns=['is_outlier', 'flagged_by'], inplace=True)
    
    outlier_report = pd.DataFrame(outlier_stats)
    
    return df_filtered, outlier_report


def apply_smoothing(
    df: pd.DataFrame,
    value_col: str = "speed",
    group_cols: List[str] = ["assay_num", "track_num"],
    window: int = 5,
    method: str = "median",
) -> pd.DataFrame:
    """Apply rolling median or mean smoothing"""
    if window <= 1:
        return df
    
    df = df.copy()
    smoothed_groups = []
    
    for group_id, group in df.groupby(group_cols):
        group = group.sort_values('time').copy()
        
        if method == "median":
            group[value_col] = group[value_col].rolling(
                window=window, center=True, min_periods=1
            ).median()
        else:
            group[value_col] = group[value_col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
        
        smoothed_groups.append(group)
    
    return pd.concat(smoothed_groups, ignore_index=True)


def convert_speed_to_mm_s(
    df: pd.DataFrame,
    pixels_per_mm: float = 104.0,
    speed_col: str = "speed",
    assume_pixels: bool = True,
) -> pd.DataFrame:
    """Convert speed from pixels/frame to mm/s"""
    df = df.copy()
    
    if speed_col in df.columns and assume_pixels:
        df[speed_col] = df[speed_col] / pixels_per_mm
    
    return df


def calculate_speed_from_xy(x, y, time, smoothing_window: int = 3, pixels_per_mm: float = 104.0):
    """Compute instantaneous speed from x,y coords in mm/s"""
    x = np.asarray(x); y = np.asarray(y); time = np.asarray(time)
    if len(x) < 2:
        return np.full_like(time, np.nan, dtype=float)

    dx = np.diff(x); dy = np.diff(y); dt = np.diff(time)
    dt = np.where(dt == 0, 1e-12, dt)
    dist_px = np.sqrt(dx**2 + dy**2)
    dist_mm = dist_px / pixels_per_mm
    v = dist_mm / dt
    v = np.r_[v[0], v]

    if smoothing_window and smoothing_window > 1:
        v = pd.Series(v).rolling(smoothing_window, center=True, min_periods=1).mean().to_numpy()
    return v


# =============================================================================
# Curve functions
# =============================================================================

def logistic_curve(t, y_initial, y_final, B, M):
    """Standard 4-parameter logistic curve (formerly Richards curve with nu=1)"""
    A = y_final - y_initial
    with np.errstate(over="ignore", invalid="ignore"):
        denom = 1.0 + np.exp(-B * (t - M))
    denom = np.clip(denom, 1e-10, 1e10)
    return y_initial + A / denom


def double_richards_curve(t, y1_i, y1_f, B1, M1, nu1, y2_f, B2, M2, nu2):
    """Double Richards curve"""
    with np.errstate(over="ignore", invalid="ignore"):
        term1 = (y1_f - y1_i) / (1.0 + np.exp(-B1 * (t - M1))) ** (1.0 / np.maximum(nu1, 1e-6))
        term2 = (y2_f - y1_f) / (1.0 + np.exp(-B2 * (t - M2))) ** (1.0 / np.maximum(nu2, 1e-6))
    term1 = np.clip(term1, -1e10, 1e10)
    term2 = np.clip(term2, -1e10, 1e10)
    return y1_i + term1 + term2


def _fit_single_bootstrap_replicate(args):
    """
    Worker function for parallel bootstrap fitting with ROBUST pointwise aggregation (median OR mean).
    Must be top-level function for multiprocessing.
    
    NEW APPROACH:
    1. Sample animals with replacement
    2. Align to common time grid
    3. Aggregate pointwise across animals (median OR mean; median is more robust to outlier animals)
    4. Fit curve to median trajectory using Huber loss (robust to outlier timepoints)
    
    Args:
        args: tuple of (bootstrap_iter, treatment, sex, genotype, sample_size,
                       animal_list, windowed_dict, trajectory_agg, window_before, window_after)
    
    Returns:
        dict with fit results and windowed data
    """
    from scipy.optimize import least_squares
    
    bootstrap_iter, treatment, sex, genotype, sample_size, animal_list, windowed_dict, trajectory_agg, window_before, window_after = args
    
    # Sample animals WITH REPLACEMENT
    sampled_indices = np.random.choice(len(animal_list), size=sample_size, replace=True)
    sampled_animals = [animal_list[i] for i in sampled_indices]
    
    # Collect windowed data for each sampled animal
    animal_dataframes = []
    for (assay, track) in sampled_animals:
        animal_window = windowed_dict[(assay, track)].copy()
        animal_dataframes.append(animal_window)
    
    # Create common time grid based on GUI window parameters (1.0s intervals to match frame rate)
    # Total duration = window_before + window_after
    total_duration = window_before + window_after
    n_points = int(total_duration / 1.0) + 1  # 1.0s interval (matches video frame rate)
    time_grid = np.linspace(-window_before, window_after, n_points)
    
    # Interpolate each animal to common grid and collect speeds
    speeds_at_each_time = []
    for animal_df in animal_dataframes:
        try:
            interpolated_speeds = np.interp(time_grid,
                                           animal_df['time_rel'].values,
                                           animal_df['speed'].values,
                                           left=np.nan, right=np.nan)
            speeds_at_each_time.append(interpolated_speeds)
        except:
            continue
    
    if len(speeds_at_each_time) == 0:
        return {
            'result': {
                'bootstrap_replicate': bootstrap_iter,
                'treatment': treatment,
                'sex': sex,
                'strain_genotype': genotype,
                'sample_size': sample_size,
                'y_initial': np.nan,
                'y_final': np.nan,
                'A': np.nan,
                'B': np.nan,
                'M': np.nan,
                'r_squared': np.nan,
                'converged': False
            },
            'windowed': None
        }
    
    # Convert to array: shape (n_animals, n_timepoints)
    speeds_array = np.array(speeds_at_each_time)
    
    # Aggregate pointwise across animals
    agg = str(trajectory_agg or 'median').strip().lower()
    if agg == 'mean':
        agg_trajectory = np.nanmean(speeds_array, axis=0)
    else:
        agg_trajectory = np.nanmedian(speeds_array, axis=0)
    
    # Remove NaN values
    valid_mask = ~np.isnan(agg_trajectory)
    if np.sum(valid_mask) < 8:
        return {
            'result': {
                'bootstrap_replicate': bootstrap_iter,
                'treatment': treatment,
                'sex': sex,
                'strain_genotype': genotype,
                'sample_size': sample_size,
                'y_initial': np.nan,
                'y_final': np.nan,
                'A': np.nan,
                'B': np.nan,
                'M': np.nan,
                'r_squared': np.nan,
                'converged': False
            },
            'windowed': None
        }
    
    t = time_grid[valid_mask]
    y = agg_trajectory[valid_mask]
    
    try:
        # Initial guess using robust percentiles
        yi_guess = np.percentile(y, 10)
        yf_guess = np.percentile(y, 90)
        p0 = [yi_guess, yf_guess, 0.1, 0.0]
        
        # Robust bounds
        y_p95 = np.percentile(y, 95)
        y_p05 = np.percentile(y, 5)
        y_range = y_p95 - y_p05
        
        lower = [max(0, y_p05 - 0.2*y_range), max(0, y_p05 - 0.2*y_range), 
                 1e-3, np.min(t) - 60]
        upper = [y_p95 + 0.2*y_range, y_p95 + 0.2*y_range, 
                 10.0, np.max(t) + 60]
        
        # Define residual function
        def residuals(params, t, y):
            return logistic_curve(t, *params) - y
        
        # Fit with HUBER loss (robust to outlier timepoints)
        result = least_squares(residuals, p0,
                              loss='huber',
                              bounds=(lower, upper),
                              max_nfev=20000,
                              args=(t, y))
        
        popt = result.x
        y_i, y_f, B, M = popt
        A = y_f - y_i
        
        y_pred = logistic_curve(t, y_i, y_f, B, M)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        converged = result.success
        
    except Exception as e:
        y_i, y_f, A, B, M = np.nan, np.nan, np.nan, np.nan, np.nan
        r2 = np.nan
        converged = False
    
    # Prepare windowed data output (median trajectory)
    wdf_out = pd.DataFrame({
        'time_rel': t,
        'speed': y,
        'bootstrap_replicate': bootstrap_iter,
        'treatment': treatment,
        'sex': sex,
        'strain_genotype': genotype
    })
    
    return {
        'result': {
            'bootstrap_replicate': bootstrap_iter,
            'treatment': treatment,
            'sex': sex,
            'strain_genotype': genotype,
            'sample_size': sample_size,
            'y_initial': y_i,
            'y_final': y_f,
            'A': A,
            'B': B,
            'M': M,
            'r_squared': r2,
            'converged': converged
        },
        'windowed': wdf_out
    }



# =============================================================================
# Windowing
# =============================================================================

def find_first_food_encounter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Find first food encounter for each animal"""
    out = []
    for (assay, track), g in df.groupby(["assay_num", "track_num"]):
        mask = (g.get("food_encounter", pd.Series(False, index=g.index)) == "food")
        if mask.any():
            first_idx = g[mask].index[0]
            out.append({
                "assay_num": assay,
                "track_num": track,
                "encounter_time": g.loc[first_idx, "time"],
                "encounter_idx": int(first_idx),
            })
    return pd.DataFrame(out)


def extract_windows(df: pd.DataFrame, window_before: float = 30.0, window_after: float = 30.0,
                    min_points: int = 10) -> Dict[Tuple[int, int], pd.DataFrame]:
    """Extract time windows around first food encounter"""
    encounters = find_first_food_encounter_rows(df)
    if encounters.empty:
        return {}

    out: Dict[Tuple[int, int], pd.DataFrame] = {}
    for _, row in encounters.iterrows():
        assay = row["assay_num"]; track = row["track_num"]; t0 = row["encounter_time"]
        g = df[(df["assay_num"] == assay) & (df["track_num"] == track)].copy()
        mask = (g["time"] >= t0 - window_before) & (g["time"] <= t0 + window_after)
        w = g.loc[mask].copy()
        if len(w) < min_points:
            continue
        w["time_rel"] = w["time"] - t0
        out[(assay, track)] = w[["time_rel", "speed"]].reset_index(drop=True)
    return out


# =============================================================================
# Fitting functions
# =============================================================================

def _failed(model_type: str, reason: str) -> dict:
    base = {
        "model_type": model_type, "fit_type": "failed", "converged": False,
        "r_squared": np.nan, "rmse": np.nan, "aic": np.nan, "bic": np.nan,
        "fitted_curve": None, "residuals": None,
        "y_initial": np.nan, "y_final": np.nan, "A": np.nan,
    }
    if model_type == "single":
        base.update({"B": np.nan, "M": np.nan})
    else:
        base.update({
            "y1_i": np.nan, "y1_f": np.nan, "y2_f": np.nan,
            "B1": np.nan, "M1": np.nan, "nu1": np.nan,
            "B2": np.nan, "M2": np.nan, "nu2": np.nan,
            "delta1": np.nan, "delta2": np.nan,
        })
    base["failure_reason"] = reason
    return base


def _metrics(y_true: np.ndarray, y_fit: np.ndarray, k_params: int) -> Tuple[float, float, float, float]:
    res = y_true - y_fit
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(res**2)))
    n = len(y_true)
    if n <= k_params or ss_res <= 0:
        return r2, rmse, np.nan, np.nan
    aic = n * np.log(ss_res / n) + 2 * k_params
    bic = n * np.log(ss_res / n) + k_params * np.log(n)
    return r2, rmse, aic, bic


def fit_single_robust(time_rel: np.ndarray, speed: np.ndarray, loss: str = 'huber') -> dict:
    """
    Fit single Richards curve using robust loss function.
    
    Args:
        time_rel: Time values relative to food encounter
        speed: Speed values  
        loss: Loss function ('least_squares', 'huber', 'soft_l1', 'cauchy')
    """
    from scipy.optimize import least_squares
    
    valid = ~(np.isnan(time_rel) | np.isnan(speed))
    t = time_rel[valid]; y = speed[valid]
    if len(t) < 8:
        return _failed("single", "insufficient_data")

    # Initial guesses using robust percentiles
    yi = np.nanpercentile(y, 10)
    yf = np.nanpercentile(y, 90)
    p0 = [yi, yf, 0.1, 0.0]
    
    # Robust bounds using percentiles
    y_p95 = np.percentile(y, 95)
    y_p05 = np.percentile(y, 5)
    y_range = y_p95 - y_p05
    
    lower = [max(0, y_p05 - 0.2*y_range), max(0, y_p05 - 0.2*y_range), 
             1e-3, np.min(t) - 60]
    upper = [y_p95 + 0.2*y_range, y_p95 + 0.2*y_range, 
             10.0, np.max(t) + 60]

    # Define residual function
    def residuals(params, t, y):
        return logistic_curve(t, *params) - y

    try:
        # Robust fitting with specified loss
        result = least_squares(residuals, p0,
                              loss=loss,
                              bounds=(lower, upper),
                              max_nfev=20000,
                              args=(t, y))
        
        params = result.x
        yhat = logistic_curve(t, *params)
        r2, rmse, aic, bic = _metrics(y, yhat, k_params=4)
        yi, yf, B, M = params
        
        return {
            "model_type": "single", "fit_type": "single", "converged": result.success,
            "r_squared": r2, "rmse": rmse, "aic": aic, "bic": bic,
            "y_initial": yi, "y_final": yf, "A": yf - yi,
            "B": B, "M": M,
            "fitted_curve": yhat, "residuals": y - yhat,
        }
    except Exception as e:
        return _failed("single", str(e))


def fit_single(time_rel: np.ndarray, speed: np.ndarray) -> dict:
    """Wrapper maintaining backward compatibility. Now uses Huber loss by default."""
    return fit_single_robust(time_rel, speed, loss='huber')


def fit_double(time_rel: np.ndarray, speed: np.ndarray) -> dict:
    valid = ~(np.isnan(time_rel) | np.isnan(speed))
    t = time_rel[valid]; y = speed[valid]
    if len(t) < 12:
        return _failed("double", "insufficient_data")

    tmin, tmax = float(t.min()), float(t.max())
    ymin, ymax = float(y.min()), float(y.max())
    ymean = float(y.mean())

    y1_i_guess = np.nanpercentile(y, 10)
    y1_f_guess = ymean
    y2_f_guess = np.nanpercentile(y, 90)
    M1_guess = tmin + 0.3 * (tmax - tmin)
    M2_guess = tmin + 0.7 * (tmax - tmin)

    p0 = [y1_i_guess, y1_f_guess, 0.2, M1_guess, 1.0, y2_f_guess, 0.2, M2_guess, 1.0]
    lower = [0.0, 0.0, 1e-3, tmin - 30, 0.1,
             0.0, 1e-3, tmin - 30, 0.1]
    upper = [ymax + 3*np.std(y), ymax + 3*np.std(y), 5.0, tmax + 30, 10.0,
             ymax + 3*np.std(y), 5.0, tmax + 30, 10.0]

    try:
        params, _ = curve_fit(double_richards_curve, t, y, p0=p0, bounds=(lower, upper),
                              maxfev=50000, method="trf")
        yhat = double_richards_curve(t, *params)
        r2, rmse, aic, bic = _metrics(y, yhat, k_params=9)
        y1_i, y1_f, B1, M1, nu1, y2_f, B2, M2, nu2 = params
        return {
            "model_type": "double", "fit_type": "double", "converged": True,
            "r_squared": r2, "rmse": rmse, "aic": aic, "bic": bic,
            "y_initial": y1_i, "y_final": y2_f, "A": y2_f - y1_i,
            "y1_i": y1_i, "y1_f": y1_f, "y2_f": y2_f,
            "delta1": y1_f - y1_i, "delta2": y2_f - y1_f,
            "B1": B1, "M1": M1, "nu1": nu1,
            "B2": B2, "M2": M2, "nu2": nu2,
            "fitted_curve": yhat, "residuals": y - yhat,
        }
    except Exception as e:
        return _failed("double", str(e))


# =============================================================================
# Plotting
# =============================================================================

def plot_summary_by_sex_strain_treatment(df_results, param, save_path):
    """Violin plot: param ~ (genotype, sex, treatment) with hierarchical ordering"""
    print(f"DEBUG: Plotting {param} to {save_path}")
    df = df_results[df_results["converged"] & df_results[param].notna()].copy()
    if df.empty:
        print(f"DEBUG: No data for {param}, returning early")
        return

    # Get hierarchically ordered groups
    ordered_group_tuples = create_hierarchical_groups(df)
    
    if not ordered_group_tuples:
        print(f"DEBUG: No groups for {param}, returning early")
        return
    
    # Extract data and labels in the correct order
    groups = []
    for treatment, sex, genotype in ordered_group_tuples:
        mask = (
            (df['treatment'] == treatment) &
            (df['sex'] == sex) &
            (df['strain_genotype'] == genotype)
        )
        group_data = df[mask][param].values
        if len(group_data) > 0:
            label = format_group_label(treatment, sex, genotype)
            groups.append((label, group_data))

    if not groups:
        print(f"DEBUG: No groups with data for {param}, returning early")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(groups)*0.8), 5))
    positions = list(range(1, len(groups) + 1))
    data_list = [g[1] for g in groups]
    labels = [g[0] for g in groups]

    parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(param)
    ax.set_title(f"{param} by Genotype | Sex | Treatment")
    
    # Set fixed y-axis range for speed parameters
    if param in ['y_initial', 'y_final']:
        ax.set_ylim(0, 0.35)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# GUI Application
# =============================================================================

class RichardsCurveFitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Richards Curve Fitting - C. elegans Speed Analysis")
        
        # Make window responsive - start with 80% of screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.85)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1000, 650)  # Minimum size
        
        # Data storage
        self.data_path = None
        self.data = None
        self.off_food_data = None  # OFF-food baseline data
        self.on_food_data = None   # ON-food encounter data
        self.off_food_path = None
        self.on_food_path = None
        self.results_single = None
        self.results_double = None
        self.outlier_report = None
        self.output_dir = None
        self.windowed_data = None
        
        # Behavioral context results
        self.behavioral_context_results = None
        self.behavioral_context_summary = None
        
        # Analysis mode for behavioral context
        # OFF-food is optional: if an OFF-food CSV is loaded, baseline/context metrics are computed; otherwise skipped
        self.context_mode = tk.StringVar(value="auto")
        
        # Outlier animal filtering (for individual mode)
        self.filter_animals_var = tk.BooleanVar(value=True)
        
        # Parameters
        self.pixels_per_mm = 104.0
        self.window_before = 30.0
        self.window_after = 30.0
        self.percentile_lower = 1.25
        self.percentile_upper = 98.75
        self.mad_threshold = 2.5
        self.smooth_window = 3
        
        self.setup_gui()
    
    def setup_gui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Configure main to have two columns and expandable rows
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)  # Make log section expandable
        
        # Title spans both columns
        ttk.Label(main, text="Richards Curve Fitting for Speed Analysis",
                  font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        
        # LEFT COLUMN
        left_frame = ttk.Frame(main)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        
        self.create_file_section(left_frame)
        self.create_parameters_section(left_frame)
        self.create_model_section(left_frame)
        
        # RIGHT COLUMN
        right_frame = ttk.Frame(main)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        
        self.create_preprocessing_section(right_frame)
        self.create_r2_flagging_section(right_frame)
        
        # Action buttons span both columns
        self.create_action_section(main)
        
        # Log section spans both columns and expands
        self.create_log_section(main)
    
    def create_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Data Files", padding="8")
        frame.pack(fill="x", pady=(0,5))
        frame.columnconfigure(1, weight=1)

        # ON-food encounter file (required)
        ttk.Label(frame, text="ON-food CSV:", font=("Helvetica", 9)).grid(row=0, column=0, sticky="w", pady=2)
        self.file_label = ttk.Label(frame, text="No file loaded", foreground="gray", font=("Helvetica", 8))
        self.file_label.grid(row=0, column=1, sticky="w", padx=3)
        ttk.Button(frame, text="Load", command=self.load_file, width=8).grid(row=0, column=2, padx=3)

        # OFF-food baseline file (optional)
        ttk.Label(frame, text="OFF-food CSV:", font=("Helvetica", 9)).grid(row=1, column=0, sticky="w", pady=2)
        self.off_food_label = ttk.Label(frame, text="Optional", foreground="gray", font=("Helvetica", 8))
        self.off_food_label.grid(row=1, column=1, sticky="w", padx=3)
        self.off_food_btn = ttk.Button(frame, text="Load", command=self.load_off_food, state='normal', width=8)
        self.off_food_btn.grid(row=1, column=2, padx=3)

        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)

        # Output directory (required)
        ttk.Label(frame, text="Output dir:", font=("Helvetica", 9)).grid(row=3, column=0, sticky="w", pady=2)
        self.output_label = ttk.Label(frame, text="Not selected", foreground="gray", font=("Helvetica", 8))
        self.output_label.grid(row=3, column=1, sticky="w", padx=3)
        ttk.Button(frame, text="Select", command=self.select_output_dir, width=8).grid(row=3, column=2, padx=3)

    def create_parameters_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding="8")
        frame.pack(fill="x", pady=(0,5))
        frame.columnconfigure(1, weight=1)
        
        # Calibration
        ttk.Label(frame, text="Pixels/mm:", font=("Helvetica", 9)).grid(row=0, column=0, sticky="w", pady=2)
        self.pixels_var = tk.DoubleVar(value=104.0)
        ttk.Entry(frame, textvariable=self.pixels_var, width=12).grid(row=0, column=1, sticky="w", padx=3)
        
        # Time windows
        ttk.Label(frame, text="Window before (s):", font=("Helvetica", 9)).grid(row=1, column=0, sticky="w", pady=2)
        self.before_var = tk.DoubleVar(value=30.0)
        ttk.Entry(frame, textvariable=self.before_var, width=12).grid(row=1, column=1, sticky="w", padx=3)
        
        ttk.Label(frame, text="Window after (s):", font=("Helvetica", 9)).grid(row=2, column=0, sticky="w", pady=2)
        self.after_var = tk.DoubleVar(value=30.0)
        ttk.Entry(frame, textvariable=self.after_var, width=12).grid(row=2, column=1, sticky="w", padx=3)
    
    def create_preprocessing_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Preprocessing", padding="8")
        frame.pack(fill="x", pady=(0,5))
        
        # Outlier filtering - more compact
        self.enable_filtering_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Outlier Filtering",
                       variable=self.enable_filtering_var,
                       command=self.toggle_filtering).pack(anchor="w", padx=5, pady=2)
        
        self.filter_frame = ttk.Frame(frame)
        self.filter_frame.pack(fill="x", padx=20, pady=2)
        
        ttk.Label(self.filter_frame, text="Lower %:", font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.lower_percentile_var = tk.DoubleVar(value=1.25)
        self.lower_entry = ttk.Entry(self.filter_frame, textvariable=self.lower_percentile_var, width=8)
        self.lower_entry.grid(row=0, column=1, padx=3)
        self.lower_entry.config(state='disabled')
        
        ttk.Label(self.filter_frame, text="Upper %:", font=("Helvetica", 8)).grid(row=0, column=2, sticky="w", padx=(5,0))
        self.upper_percentile_var = tk.DoubleVar(value=98.75)
        self.upper_entry = ttk.Entry(self.filter_frame, textvariable=self.upper_percentile_var, width=8)
        self.upper_entry.grid(row=0, column=3, padx=3)
        self.upper_entry.config(state='disabled')
        
        ttk.Label(self.filter_frame, text="MAD:", font=("Helvetica", 8)).grid(row=1, column=0, sticky="w")
        self.mad_threshold_var = tk.DoubleVar(value=2.5)
        self.mad_entry = ttk.Entry(self.filter_frame, textvariable=self.mad_threshold_var, width=8)
        self.mad_entry.grid(row=1, column=1, padx=3)
        self.mad_entry.config(state='disabled')
        
        # Smoothing - more compact
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=5)
        
        self.enable_smoothing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Smoothing",
                       variable=self.enable_smoothing_var,
                       command=self.toggle_smoothing).pack(anchor="w", padx=5, pady=2)
        
        self.smooth_frame = ttk.Frame(frame)
        self.smooth_frame.pack(fill="x", padx=20, pady=2)
        
        ttk.Label(self.smooth_frame, text="Window:", font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.smooth_window_var = tk.IntVar(value=3)
        self.smooth_window_entry = ttk.Entry(self.smooth_frame, textvariable=self.smooth_window_var, width=8)
        self.smooth_window_entry.grid(row=0, column=1, padx=3)
        self.smooth_window_entry.config(state='normal')
        
        ttk.Label(self.smooth_frame, text="Method:", font=("Helvetica", 8)).grid(row=0, column=2, sticky="w", padx=(5,0))
        self.smooth_method_var = tk.StringVar(value="mean")
        method_frame = ttk.Frame(self.smooth_frame)
        method_frame.grid(row=0, column=3, sticky="w", padx=3)
        self.median_radio = ttk.Radiobutton(method_frame, text="Median", variable=self.smooth_method_var, 
                                           value="median", state='disabled')
        self.median_radio.pack(side="left", padx=2)
        self.mean_radio = ttk.Radiobutton(method_frame, text="Mean", variable=self.smooth_method_var, 
                                         value="mean", state='disabled')
        self.mean_radio.pack(side="left", padx=2)
    
    def create_r2_flagging_section(self, parent):
        frame = ttk.LabelFrame(parent, text="R² Quality Flagging", padding="8")
        frame.pack(fill="x", pady=(0,5))
        
        # R² flagging checkbox
        self.enable_r2_flagging_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Flag Poor Fits",
                       variable=self.enable_r2_flagging_var,
                       command=self.toggle_r2_flagging).pack(anchor="w", padx=5, pady=2)
        
        self.r2_frame = ttk.Frame(frame)
        self.r2_frame.pack(fill="x", padx=20, pady=2)
        
        ttk.Label(self.r2_frame, text="R² threshold:", font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.r2_threshold_var = tk.DoubleVar(value=0.7)
        self.r2_entry = ttk.Entry(self.r2_frame, textvariable=self.r2_threshold_var, width=8)
        self.r2_entry.grid(row=0, column=1, padx=3)
        self.r2_entry.config(state='disabled')
        
        ttk.Label(self.r2_frame, text="(Below = flagged)",
                 font=("Helvetica", 7, "italic"), foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w")
    
    def toggle_r2_flagging(self):
        state = 'normal' if self.enable_r2_flagging_var.get() else 'disabled'
        self.r2_entry.config(state=state)
    
    def create_model_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Model Selection", padding="8")
        frame.pack(fill="x", pady=(0,5))
        
        self.model_var = tk.StringVar(value="bootstrap")
        ttk.Radiobutton(frame, text="Individual Fits", variable=self.model_var, 
                       value="single").pack(anchor="w", padx=5, pady=1)
        
        # Individual mode option
        individual_frame = ttk.Frame(frame)
        individual_frame.pack(anchor="w", padx=20, pady=3)
        ttk.Checkbutton(individual_frame, text="Filter outlier animals (y_initial & baseline_speed)",
                       variable=self.filter_animals_var,
                       ).pack(anchor="w")
        
        ttk.Radiobutton(frame, text="Bootstrap Aggregation", variable=self.model_var, 
                       value="bootstrap").pack(anchor="w", padx=5, pady=1)
        
        # Bootstrap options - more compact
        bootstrap_frame = ttk.Frame(frame)
        bootstrap_frame.pack(anchor="w", padx=20, pady=3)
        
        ttk.Label(bootstrap_frame, text="Replicates:", font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.n_bootstrap_var = tk.IntVar(value=5000)
        ttk.Entry(bootstrap_frame, textvariable=self.n_bootstrap_var, width=8).grid(row=0, column=1, padx=3)
        # Sample size is automatically set to N animals per group (classical bootstrap)

        ttk.Label(bootstrap_frame, text="Pointwise agg:", font=("Helvetica", 8)).grid(row=2, column=0, sticky="w")
        self.bootstrap_pointwise_agg_var = tk.StringVar(value="median")
        ttk.Combobox(bootstrap_frame, textvariable=self.bootstrap_pointwise_agg_var,
                     values=["median", "mean"], state="readonly", width=8).grid(row=2, column=1, padx=3)

    def create_action_section(self, parent):
        frame = ttk.Frame(parent, padding="5")
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.run_btn = ttk.Button(frame, text="Run Analysis", command=self.run_analysis,
                                  style='Accent.TButton')
        self.run_btn.pack(side="left", padx=5)
        self.run_btn.config(state='disabled')
        
        ttk.Button(frame, text="Save Results", command=self.save_results).pack(side="left", padx=5)
        ttk.Button(frame, text="Save Permutation Package", 
                  command=self.save_permutation_package).pack(side="left", padx=5)
        ttk.Button(frame, text="Clear Log", command=self.clear_log).pack(side="left", padx=5)
        
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
    
    def create_log_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Log", padding="5")
        frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=5)
        
        self.log_text = scrolledtext.ScrolledText(frame, height=12, wrap=tk.WORD, font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True)
    
    def toggle_filtering(self):
        state = 'normal' if self.enable_filtering_var.get() else 'disabled'
        self.lower_entry.config(state=state)
        self.upper_entry.config(state=state)
        self.mad_entry.config(state=state)
    
    def toggle_smoothing(self):
        state = 'normal' if self.enable_smoothing_var.get() else 'disabled'
        self.smooth_window_entry.config(state=state)
        self.median_radio.config(state=state)
        self.mean_radio.config(state=state)
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.data = pd.read_csv(filename)
                self.data_path = Path(filename)
                
                # Treat the primary loaded file as ON-food encounter data
                self.on_food_data = self.data
                self.on_food_path = self.data_path
                self.file_label.config(text=f"ON-food: {self.data_path.name}", foreground="black")
                
                self.update_run_button_state()
                self.log(f"Loaded: {filename}")
                self.log(f"  Rows: {len(self.data)}")
                self.log(f"  Columns: {', '.join(self.data.columns)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.log(f"ERROR loading file: {str(e)}")
    
    def load_off_food(self):
        filename = filedialog.askopenfilename(
            title="Select OFF-food baseline CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.off_food_data = pd.read_csv(filename)
                self.off_food_path = Path(filename)
                self.off_food_label.config(text=f"OFF-food: {self.off_food_path.name}", foreground="black")
                self.update_run_button_state()
                self.log(f"Loaded OFF-food baseline: {filename}")
                self.log(f"  Rows: {len(self.off_food_data)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load OFF-food file:\n{str(e)}")
                self.log(f"ERROR loading OFF-food file: {str(e)}")
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.output_label.config(text=self.output_dir.name, foreground="black")
            self.update_run_button_state()
            self.log(f"Output directory: {directory}")
    
    def update_run_button_state(self):
        # OFF-food is optional; require ON-food data and an output directory
        ready = (self.data is not None and self.output_dir is not None)

        if ready:
            self.run_btn.config(state='normal')
        else:
            self.run_btn.config(state='disabled')

    def run_analysis(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return
        
        if self.output_dir is None:
            messagebox.showwarning("Warning", "Please select an output directory first")
            return
        
        try:
            self.progress.start()
            self.run_btn.config(state='disabled')
            
            self.log("\n" + "="*70)
            self.log("STARTING RICHARDS CURVE FITTING ANALYSIS")
            self.log("="*70)
            
            # Get parameters
            self.pixels_per_mm = self.pixels_var.get()
            self.window_before = self.before_var.get()
            self.window_after = self.after_var.get()
            
            df = self.data.copy()
            
            # Preprocessing
            self.log("\n" + "="*70)
            self.log("PREPROCESSING")
            self.log("="*70)
            
            # Step 1: Ensure speed exists
            if "speed" not in df.columns and {"x","y","time"}.issubset(df.columns):
                self.log("Computing speed from x,y coordinates...")
                self.root.update_idletasks()  # Update GUI
                df["speed"] = calculate_speed_from_xy(
                    df["x"].to_numpy(), 
                    df["y"].to_numpy(),
                    df["time"].to_numpy(), 
                    smoothing_window=3,
                    pixels_per_mm=self.pixels_per_mm
                )
                self.log("  Speed computed in mm/s")
            elif "speed" in df.columns:
                self.log("Converting existing speed column to mm/s...")
                self.root.update_idletasks()  # Update GUI
                df = convert_speed_to_mm_s(df, pixels_per_mm=self.pixels_per_mm, assume_pixels=True)
                self.log("  Conversion complete")
            
            self.root.update_idletasks()  # Update GUI
            
            # Step 1.5: Hard speed threshold filter (always applied)
            self.log("\nApplying hard speed threshold filter...")
            self.log(f"  Removing speeds > 0.4 mm/s (biologically implausible)")
            df = apply_speed_threshold_filter(df, max_speed=0.4)
            
            self.root.update_idletasks()  # Update GUI
            
            # Step 2: Outlier filtering
            if self.enable_filtering_var.get():
                self.log("\nApplying outlier filtering...")
                self.root.update_idletasks()  # Update GUI
                self.percentile_lower = self.lower_percentile_var.get()
                self.percentile_upper = self.upper_percentile_var.get()
                self.mad_threshold = self.mad_threshold_var.get()
                
                self.log(f"  Percentile bounds: {self.percentile_lower}% - {self.percentile_upper}%")
                self.log(f"  MAD threshold: {self.mad_threshold}x")
                
                n_before = len(df)
                df, self.outlier_report = apply_hybrid_outlier_filter(
                    df,
                    percentile_lower=self.percentile_lower,
                    percentile_upper=self.percentile_upper,
                    mad_threshold=self.mad_threshold,
                )
                n_after = len(df)
                n_removed = n_before - n_after
                pct_removed = (n_removed / n_before * 100) if n_before > 0 else 0
                
                self.log(f"  Removed {n_removed} outlier points ({pct_removed:.2f}%)")
                self.log(f"  Remaining: {n_after} points")
                self.root.update_idletasks()  # Update GUI
            
            # Step 3: Smoothing
            if self.enable_smoothing_var.get():
                self.log("\nApplying smoothing...")
                self.root.update_idletasks()  # Update GUI
                self.smooth_window = self.smooth_window_var.get()
                smooth_method = self.smooth_method_var.get()
                
                self.log(f"  Method: {smooth_method}")
                self.log(f"  Window: {self.smooth_window} frames")
                
                df = apply_smoothing(
                    df,
                    window=self.smooth_window,
                    method=smooth_method,
                )
                self.log("  Smoothing complete")
                self.root.update_idletasks()  # Update GUI
            
            # Extract windows
            self.log("\n" + "="*70)
            self.log("EXTRACTING WINDOWS AROUND FOOD ENCOUNTERS")
            self.log("="*70)
            self.root.update_idletasks()  # Update GUI
            
            windowed = extract_windows(df, 
                                      window_before=self.window_before,
                                      window_after=self.window_after,
                                      min_points=10)
            
            if not windowed:
                messagebox.showerror("Error", "No valid encounter windows found")
                self.log("ERROR: No encounter windows found")
                return
            
            self.log(f"Found {len(windowed)} animals with valid windows")
            self.root.update_idletasks()  # Update GUI
            
            # Process OFF-food baseline data first if an OFF-food file was loaded
            if self.off_food_data is not None:
                self.log("\n" + "="*70)
                self.log("PROCESSING OFF-FOOD BASELINE DATA")
                self.log("="*70)
                
                # Process OFF-food data through same pipeline
                df_off = self.off_food_data.copy()
                
                # Apply same preprocessing
                if "speed" not in df_off.columns and {"x","y","time"}.issubset(df_off.columns):
                    self.log("Computing speed from x,y coordinates...")
                    df_off["speed"] = calculate_speed_from_xy(
                        df_off["x"].to_numpy(), 
                        df_off["y"].to_numpy(),
                        df_off["time"].to_numpy(), 
                        smoothing_window=3,
                        pixels_per_mm=self.pixels_per_mm
                    )
                elif "speed" in df_off.columns:
                    df_off = convert_speed_to_mm_s(df_off, pixels_per_mm=self.pixels_per_mm, assume_pixels=True)
                
                # Hard speed threshold filter (always applied)
                self.log("Applying hard speed threshold to OFF-food data...")
                self.log(f"  Removing speeds > 0.4 mm/s")
                df_off = apply_speed_threshold_filter(df_off, max_speed=0.4)
                
                if self.enable_filtering_var.get():
                    self.log("Applying outlier filtering to OFF-food data...")
                    df_off, _ = apply_hybrid_outlier_filter(
                        df_off,
                        percentile_lower=self.percentile_lower,
                        percentile_upper=self.percentile_upper,
                        mad_threshold=self.mad_threshold,
                    )
                
                if self.enable_smoothing_var.get():
                    self.log("Applying smoothing to OFF-food data...")
                    df_off = apply_smoothing(
                        df_off,
                        window=self.smooth_window,
                        method=self.smooth_method_var.get(),
                    )
                
                # Extract windows for OFF-food (same time window as ON-food)
                self.log("\nExtracting time windows from OFF-food data...")
                self.log(f"  Using same window as ON-food: -{self.window_before}s to +{self.window_after}s")
                
                windowed_off = extract_windows(df_off, 
                                               window_before=self.window_before,
                                               window_after=self.window_after,
                                               min_points=10)
                
                if not windowed_off:
                    self.log("ERROR: No valid windows found in OFF-food data")
                    messagebox.showerror("Error", "No valid windows in OFF-food data. Check that food_encounter column exists.")
                    return
                
                self.log(f"Found {len(windowed_off)} OFF-food animals with valid windows")
                
                # Bootstrap OFF-food baselines (parallel to ON-food bootstrap)
                self.log("\nBootstrapping OFF-food baseline speeds...")
                
                # Group OFF-food animals by treatment × sex × genotype
                off_animal_groups = {}
                for (assay, track), wdf in windowed_off.items():
                    animal_meta = df_off[(df_off['assay_num']==assay) & (df_off['track_num']==track)].iloc[0]
                    treatment = animal_meta['treatment'] if 'treatment' in df_off.columns else 'unknown'
                    sex = animal_meta['sex'] if 'sex' in df_off.columns else 'unknown'
                    genotype = animal_meta['strain_genotype'] if 'strain_genotype' in df_off.columns else 'unknown'
                    
                    group_key = (treatment, sex, genotype)
                    if group_key not in off_animal_groups:
                        off_animal_groups[group_key] = []
                    off_animal_groups[group_key].append((assay, track))
                
                self.log(f"Found {len(off_animal_groups)} OFF-food groups")
                for group_key, animals in off_animal_groups.items():
                    self.log(f"  {group_key[0]} | {group_key[1]} | {group_key[2]}: {len(animals)} animals")
                
                # Process OFF-food based on model mode
                model_choice = self.model_var.get()
                
                if model_choice == "single":
                    # INDIVIDUAL MODE: Simple mean per animal
                    self.log("\nCalculating individual OFF-food baselines (mean per animal)...")
                    
                    off_food_individual_results = []
                    for (assay, track), wdf in windowed_off.items():
                        animal_meta = df_off[(df_off['assay_num']==assay) & (df_off['track_num']==track)].iloc[0]
                        treatment = animal_meta['treatment'] if 'treatment' in df_off.columns else 'unknown'
                        sex = animal_meta['sex'] if 'sex' in df_off.columns else 'unknown'
                        genotype = animal_meta['strain_genotype'] if 'strain_genotype' in df_off.columns else 'unknown'
                        
                        # Simple mean speed for this animal
                        mean_speed = wdf['speed'].mean()
                        
                        off_food_individual_results.append({
                            'assay_num': assay,
                            'track_num': track,
                            'treatment': treatment,
                            'sex': sex,
                            'strain_genotype': genotype,
                            'baseline_speed': mean_speed
                        })
                    
                    self.off_food_results = pd.DataFrame(off_food_individual_results)
                    self.log(f"Calculated {len(self.off_food_results)} individual OFF-food baselines")
                    
                    # Filter outlier OFF-food animals by baseline_speed (OPTIONAL)
                    if self.filter_animals_var.get():
                        self.log("\nFiltering outlier OFF-food animals by baseline_speed...")
                        self.off_food_results = filter_outlier_animals_by_parameter(
                            self.off_food_results,
                            parameter='baseline_speed',
                            percentile_lower=self.percentile_lower,
                            percentile_upper=self.percentile_upper,
                            mad_threshold=self.mad_threshold,
                            group_cols=['treatment', 'sex', 'strain_genotype']
                        )
                    else:
                        self.log("\nSkipping outlier animal filtering (disabled by user)")
                    
                    # Save individual OFF-food results
                    off_dir = self.output_dir / "off_food_baselines"
                    off_dir.mkdir(exist_ok=True)
                    self.off_food_results.to_csv(off_dir / "off_food_individual_baselines.csv", index=False)
                    self.log(f"Saved OFF-food individual results to: off_food_individual_baselines.csv")
                    
                elif model_choice == "bootstrap":
                    # BOOTSTRAP MODE: Bootstrap sampling
                    self.log("\nBootstrapping OFF-food baseline speeds...")
                    
                    n_bootstrap = self.n_bootstrap_var.get()
                    
                    # FIRST: Calculate and save individual animal baselines for permutation testing
                    self.log("\nCalculating individual animal OFF-food baselines...")
                    individual_baselines = []
                    
                    for group_key, animal_list in off_animal_groups.items():
                        treatment, sex, genotype = group_key
                        
                        for (assay, track) in animal_list:
                            window_data = windowed_off[(assay, track)]
                            mean_speed = window_data['speed'].mean()
                            
                            individual_baselines.append({
                                'assay_num': assay,
                                'track_num': track,
                                'baseline_speed': mean_speed,
                                'treatment': treatment,
                                'sex': sex,
                                'strain_genotype': genotype
                            })
                    
                    individual_df = pd.DataFrame(individual_baselines)
                    
                    # Apply outlier filtering if enabled (same as single mode)
                    if self.filter_animals_var.get():
                        self.log("\nFiltering outlier OFF-food animals by baseline_speed...")
                        individual_df = filter_outlier_animals_by_parameter(
                            individual_df,
                            parameter='baseline_speed',
                            percentile_lower=self.percentile_lower,
                            percentile_upper=self.percentile_upper,
                            mad_threshold=self.mad_threshold,
                            group_cols=['treatment', 'sex', 'strain_genotype']
                        )
                    else:
                        self.log("\nSkipping outlier animal filtering (disabled by user)")
                    
                    # Save individual baselines for permutation testing
                    off_dir = self.output_dir / "off_food_baselines"
                    off_dir.mkdir(exist_ok=True)
                    individual_df.to_csv(off_dir / "off_food_individual_baselines.csv", index=False)
                    self.log(f"Saved {len(individual_df)} individual OFF-food baselines to: off_food_individual_baselines.csv")
                    
                    # SECOND: Perform bootstrap aggregation
                    self.log("\nPerforming bootstrap aggregation on OFF-food baselines...")
                    off_food_bootstrap_results = []
                    off_food_windowed_list = []  # aggregated trace per replicate

                    # Trajectory aggregation — mirrors ON-food bootstrap setting
                    off_traj_agg = str(self.bootstrap_pointwise_agg_var.get()).strip().lower()
                    if off_traj_agg not in ('median', 'mean'):
                        off_traj_agg = 'median'

                    # Common time grid — same 1s resolution as ON-food bootstrap worker
                    n_grid_points = int((self.window_before + self.window_after) / 1.0) + 1
                    off_time_grid = np.linspace(-self.window_before, self.window_after, n_grid_points)

                    for group_key, animal_list in off_animal_groups.items():
                        treatment, sex, genotype = group_key
                        n_animals = len(animal_list)
                        # Classical bootstrap: each replicate resamples N_group animals with replacement
                        actual_sample_size = n_animals
                        self.log(f"  Using classical bootstrap for OFF-food {group_key}: resampling N_group={n_animals} animals with replacement each replicate")

                        self.log(f"  Bootstrapping OFF-food {group_key}: {n_bootstrap} replicates (vectorized)...")

                        # ── Precompute per-animal quantities ONCE, outside the bootstrap loop ──
                        # (identical math to the original per-iteration calls — just no longer
                        #  recomputed redundantly on every one of the n_bootstrap replicates)
                        scalar_means = np.empty(n_animals, dtype=np.float64)
                        trace_matrix = np.full((n_animals, n_grid_points), np.nan, dtype=np.float64)

                        for i, (assay, track) in enumerate(animal_list):
                            window_data = windowed_off[(assay, track)]
                            scalar_means[i] = window_data['speed'].mean()
                            try:
                                trace_matrix[i, :] = np.interp(
                                    off_time_grid,
                                    window_data['time_rel'].values,
                                    window_data['speed'].values,
                                    left=np.nan, right=np.nan)
                            except Exception:
                                # Same effect as the original's per-iteration "continue":
                                # this animal contributes nothing (stays all-NaN) wherever it's sampled.
                                pass

                        # ── Draw ALL bootstrap resamples for this group in a single call ──
                        # (statistically identical sampling — animals chosen with replacement,
                        #  same N_group per replicate — just generated in one batch instead of
                        #  one np.random.choice call per replicate)
                        sample_idx = np.random.choice(n_animals, size=(n_bootstrap, actual_sample_size), replace=True)

                        # ── Scalar baselines: mean across sampled animals, vectorized over all replicates ──
                        bootstrap_means = scalar_means[sample_idx].mean(axis=1)  # shape (n_bootstrap,)

                        off_food_bootstrap_results.extend(
                            {
                                'bootstrap_replicate': it,
                                'treatment': treatment,
                                'sex': sex,
                                'strain_genotype': genotype,
                                'sample_size': actual_sample_size,
                                'baseline_speed': float(bootstrap_means[it]),
                            }
                            for it in range(n_bootstrap)
                        )

                        # ── Aggregated windowed traces, vectorized and processed in memory-bounded chunks ──
                        # Chunking only affects how much RAM is used at once; the math per replicate
                        # (nanmean/nanmedian across sampled animals at each grid point, then drop
                        # still-NaN points) is identical to the original per-replicate computation.
                        elems_per_replicate = max(1, actual_sample_size * n_grid_points)
                        chunk_size = max(1, min(n_bootstrap, int(2_000_000 // elems_per_replicate) or 1))

                        for chunk_start in range(0, n_bootstrap, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, n_bootstrap)
                            idx_chunk = sample_idx[chunk_start:chunk_end]            # (chunk, n_animals)
                            sampled_traces = trace_matrix[idx_chunk]                 # (chunk, n_animals, n_grid)

                            with np.errstate(invalid='ignore'):
                                if off_traj_agg == 'mean':
                                    agg_chunk = np.nanmean(sampled_traces, axis=1)   # (chunk, n_grid)
                                else:
                                    agg_chunk = np.nanmedian(sampled_traces, axis=1)

                            valid_chunk = ~np.isnan(agg_chunk)
                            if not valid_chunk.any():
                                continue

                            n_rep_chunk = chunk_end - chunk_start
                            rep_ids = np.arange(chunk_start, chunk_end)
                            bootstrap_idx_grid = np.repeat(rep_ids, n_grid_points)
                            time_grid_tiled = np.tile(off_time_grid, n_rep_chunk)
                            valid_flat = valid_chunk.ravel()

                            off_food_windowed_list.append(pd.DataFrame({
                                'time_rel':            time_grid_tiled[valid_flat],
                                'speed':               agg_chunk.ravel()[valid_flat],
                                'bootstrap_replicate': bootstrap_idx_grid[valid_flat],
                                'treatment':           treatment,
                                'sex':                 sex,
                                'strain_genotype':     genotype,
                            }))

                    self.off_food_results = pd.DataFrame(off_food_bootstrap_results)
                    self.log(f"Calculated {len(self.off_food_results)} OFF-food bootstrap replicates")
                    
                    # Save OFF-food bootstrap baseline results (scalar mean per replicate)
                    self.off_food_results.to_csv(off_dir / "off_food_bootstrap_baselines.csv", index=False)
                    self.log(f"Saved OFF-food bootstrap results to: off_food_bootstrap_baselines.csv")

                    # Save aggregated windowed traces — same format as windowed_speed_data.csv
                    # columns: time_rel, speed, bootstrap_replicate, treatment, sex, strain_genotype
                    if off_food_windowed_list:
                        df_off_windowed = pd.concat(off_food_windowed_list, ignore_index=True)
                        df_off_windowed.to_csv(off_dir / "off_food_windowed_speed_data.csv", index=False)
                        self.log(f"Saved OFF-food windowed traces to: off_food_windowed_speed_data.csv ({len(df_off_windowed)} rows)")
                    else:
                        self.log("WARNING: No OFF-food windowed traces were generated")
            
            # Fit models
            model_choice = self.model_var.get()
            
            if model_choice == "single":
                self.log("\n" + "="*70)
                self.log("FITTING SINGLE RICHARDS MODEL (INDIVIDUAL FITS)")
                self.log("="*70)
                self.results_single = self.fit_model(df, windowed, "single")
                self.report_model_summary(self.results_single, "single")
            
            elif model_choice == "bootstrap":
                self.log("\n" + "="*70)
                self.log("FITTING BOOTSTRAP AGGREGATION MODEL")
                self.log("="*70)
                n_bootstrap = self.n_bootstrap_var.get()
                self.log(f"Bootstrap replicates: {n_bootstrap}")
                self.log("Bootstrap sample size: N animals per group (classical bootstrap; N varies by group)")
                self.log(f"Time window: -{self.window_before}s to +{self.window_after}s around food encounter")
                self.results_single = self.fit_bootstrap_model(df, windowed, n_bootstrap)
                self.report_model_summary(self.results_single, "bootstrap")
            
            # Behavioral Context Analysis (optional: requires OFF-food baseline)
            if hasattr(self, 'off_food_results') and self.off_food_results is not None:
                self.calculate_behavioral_context()
            else:
                self.log('\nSkipping behavioral context analysis (no OFF-food baseline loaded)')
            self.log("\n" + "="*70)
            self.log("ANALYSIS COMPLETE")
            self.log("="*70)
            
            messagebox.showinfo("Success", "Analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.log(f"\nERROR: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            self.progress.stop()
            self.run_btn.config(state='normal')
    
    def fit_model(self, df_master, windowed, model_type):
        """Fit model and save all outputs to disk"""
        # Create output directory structure (clear if exists to ensure fresh outputs)
        model_dir = self.output_dir / f"model_{model_type}"
        
        # Remove existing directory if it exists to start fresh
        if model_dir.exists():
            shutil.rmtree(model_dir)
            self.log(f"  Cleared existing {model_type} model directory")
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        windowed_data_list = []
        
        total = len(windowed)
        for i, ((assay, track), wdf) in enumerate(windowed.items()):
            if i % 5 == 0:  # Update more frequently
                self.log(f"  Processing animal {i+1}/{total}...")
                self.root.update_idletasks()  # Update GUI
            
            t_rel = wdf["time_rel"].to_numpy()
            spd = wdf["speed"].to_numpy()
            
            if model_type == "single":
                res = fit_single(t_rel, spd)
            else:
                res = fit_double(t_rel, spd)
            
            # Add metadata
            animal_meta = df_master[(df_master["assay_num"] == assay) & 
                                   (df_master["track_num"] == track)].iloc[0]
            res["assay_num"] = assay
            res["track_num"] = track
            for col in ["sex", "strain_genotype", "treatment"]:
                res[col] = animal_meta.get(col, np.nan)
            
            results.append(res)
            
            # Store windowed data
            wdf_out = wdf.copy()
            wdf_out["assay_num"] = assay
            wdf_out["track_num"] = track
            windowed_data_list.append(wdf_out)
        
        self.root.update_idletasks()  # Update GUI
        
        df_results = pd.DataFrame(results)
        df_windowed = pd.concat(windowed_data_list, ignore_index=True)
        
        # Filter outlier animals by y_initial (for individual mode only, OPTIONAL)
        if model_type == "single" and self.filter_animals_var.get():
            self.log("\nFiltering outlier ON-food animals by y_initial...")
            df_results = filter_outlier_animals_by_parameter(
                df_results,
                parameter='y_initial',
                percentile_lower=self.percentile_lower,
                percentile_upper=self.percentile_upper,
                mad_threshold=self.mad_threshold,
                group_cols=['treatment', 'sex', 'strain_genotype']
            )
        elif model_type == "single":
            self.log("\nSkipping outlier animal filtering (disabled by user)")
        
        # Add behavioral context if OFF-food data available (for individual mode)
        if model_type == "single" and hasattr(self, 'off_food_results') and self.off_food_results is not None:
            self.log("\nAdding behavioral context scores for individual animals...")
            df_results = self.add_behavioral_context(df_results, "single")
            self.log(f"  Added food_detection_score and food_encounter_score columns")
        
        # Save CSV files with correct names for viewer compatibility
        df_results.to_csv(model_dir / "richards_fit_parameters.csv", index=False)
        df_windowed.to_csv(model_dir / "windowed_speed_data.csv", index=False)  # Correct filename!
        
        # For individual mode, ALSO save windowed_speed_data_original.csv (same format as bootstrap)
        # This allows the coverage analyzer to work with individual mode data
        if model_type == "single":
            self.log(f"\nSaving windowed_speed_data_original.csv for coverage analysis...")
            windowed_original_list = []
            for (assay, track), wdf in windowed.items():
                # Get metadata for this animal
                animal_meta = df_master[(df_master['assay_num']==assay) & (df_master['track_num']==track)].iloc[0]
                
                # Add identifiers and metadata
                wdf_out = wdf.copy()
                wdf_out['assay_num'] = assay
                wdf_out['track_num'] = track
                wdf_out['treatment'] = animal_meta['treatment'] if 'treatment' in df_master.columns else 'unknown'
                wdf_out['sex'] = animal_meta['sex'] if 'sex' in df_master.columns else 'unknown'
                wdf_out['strain_genotype'] = animal_meta['strain_genotype'] if 'strain_genotype' in df_master.columns else 'unknown'
                
                windowed_original_list.append(wdf_out)
            
            df_windowed_original = pd.concat(windowed_original_list, ignore_index=True)
            df_windowed_original.to_csv(model_dir / "windowed_speed_data_original.csv", index=False)
            self.log(f"  Saved {len(windowed)} individual animals to windowed_speed_data_original.csv")
        
        self.log(f"  Saved results to {model_dir}")
        
        # Generate parameter group summaries
        self.log(f"  Generating parameter group summaries...")
        self.generate_parameter_summaries(df_results, model_dir, model_type)
        
        self.root.update_idletasks()  # Update GUI
        
        # Generate parameter cheat sheet
        self.save_parameter_cheatsheet(model_dir, model_type)
        
        # Generate summary plots
        self.log(f"  Generating summary plots...")
        self.root.update_idletasks()  # Update GUI
        
        # Add behavioral context scores to metrics if available
        metrics = (["y_initial","y_final","A","B","M"] if model_type=="single"
                   else ["y_initial","y_final","A","y1_i","y1_f","y2_f","delta1","delta2","B1","M1","nu1","B2","M2","nu2"])
        
        if 'food_detection_score' in df_results.columns:
            metrics.extend(["food_detection_score", "food_encounter_score"])
        
        plot_count = 0
        for m in metrics:
            try:
                plot_summary_by_sex_strain_treatment(df_results, m, model_dir / f"{m}_by_sex_strain_treatment.png")
                plot_count += 1
            except Exception as e:
                self.log(f"    Warning: Could not generate plot for {m}: {str(e)}")
        
        self.log(f"  Generated {plot_count}/{len(metrics)} summary plots")
        
        # Generate R² distribution histogram with threshold line
        self.plot_r2_distribution(df_results, model_dir, model_type)
        
        self.root.update_idletasks()  # Update GUI
        return df_results
    
    def fit_bootstrap_model(self, df_master, windowed, n_bootstrap):
        """
        Fit Richards curves using bootstrap sampling of animals
        Returns DataFrame with all bootstrap replicates
        """
        model_dir = self.output_dir / "bootstrap_model"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Clear directory
        for item in model_dir.glob('*'):
            if item.is_file():
                item.unlink()
        
        # Group animals by treatment × sex × genotype
        animal_groups = {}
        for (assay, track), wdf in windowed.items():
            # Get metadata
            animal_meta = df_master[(df_master['assay_num']==assay) & (df_master['track_num']==track)].iloc[0]
            treatment = animal_meta['treatment'] if 'treatment' in df_master.columns else 'unknown'
            sex = animal_meta['sex'] if 'sex' in df_master.columns else 'unknown'
            genotype = animal_meta['strain_genotype'] if 'strain_genotype' in df_master.columns else 'unknown'
            
            group_key = (treatment, sex, genotype)
            if group_key not in animal_groups:
                animal_groups[group_key] = []
            animal_groups[group_key].append((assay, track))
        
        self.log(f"\nFound {len(animal_groups)} groups for bootstrap sampling")
        for group_key, animals in animal_groups.items():
            self.log(f"  {group_key[0]} | {group_key[1]} | {group_key[2]}: {len(animals)} animals")
        
        # Save ORIGINAL windowed data (individual animals, not bootstrap replicates)
        # This is needed for permutation testing
        self.log(f"\nSaving original windowed animal data for permutation testing...")
        windowed_original_list = []
        for (assay, track), wdf in windowed.items():
            # Get metadata for this animal
            animal_meta = df_master[(df_master['assay_num']==assay) & (df_master['track_num']==track)].iloc[0]
            
            # Add identifiers and metadata
            wdf_out = wdf.copy()
            wdf_out['assay_num'] = assay
            wdf_out['track_num'] = track
            wdf_out['treatment'] = animal_meta['treatment'] if 'treatment' in df_master.columns else 'unknown'
            wdf_out['sex'] = animal_meta['sex'] if 'sex' in df_master.columns else 'unknown'
            wdf_out['strain_genotype'] = animal_meta['strain_genotype'] if 'strain_genotype' in df_master.columns else 'unknown'
            
            windowed_original_list.append(wdf_out)
        
        df_windowed_original = pd.concat(windowed_original_list, ignore_index=True)
        df_windowed_original.to_csv(model_dir / "windowed_speed_data_original.csv", index=False)
        self.log(f"  Saved {len(windowed)} individual animals to windowed_speed_data_original.csv")
        
        # Bootstrap sampling
        results = []
        windowed_data_list = []
        total_fits = n_bootstrap * len(animal_groups)
        fit_count = 0
        
        self.log(f"\nPerforming bootstrap sampling ({total_fits} total fits)...")
        
        # Switch to determinate progress bar
        self.progress.config(mode='determinate', maximum=total_fits, value=0)
        self.root.update()  # Force GUI to show the message and progress bar
        
        # Determine number of CPU cores to use (conservative: 75% of available cores)
        max_cores = cpu_count()
        # Use 75% of cores, leaving 25% free for other tasks
        n_cores = max(1, int(max_cores * 0.75))
        self.log(f"Using {n_cores} of {max_cores} CPU cores for parallel processing")
        self.log(f"  (This leaves {max_cores - n_cores} cores for other tasks)")
        
        for group_key, animal_list in animal_groups.items():
            treatment, sex, genotype = group_key
            n_animals = len(animal_list)
            # Classical bootstrap: each replicate resamples N_group animals with replacement
            # (i.e., the bootstrap sample size equals the number of unique animals in the group)
            actual_sample_size = n_animals
            self.log(f"  Using classical bootstrap for {group_key}: resampling N_group={n_animals} animals with replacement each replicate")
            
            self.log(f"\n  Bootstrapping {group_key}...")
            self.log(f"    Starting {n_bootstrap} replicates using {n_cores} cores...")
            self.root.update()  # Force complete GUI update
            
            # Prepare arguments for parallel processing
            tasks = []
            trajectory_agg = str(self.bootstrap_pointwise_agg_var.get()).strip().lower()
            if trajectory_agg not in ('median', 'mean'):
                trajectory_agg = 'median'
            for bootstrap_iter in range(n_bootstrap):
                tasks.append((bootstrap_iter, treatment, sex, genotype, actual_sample_size, 
                             animal_list, windowed, trajectory_agg, self.window_before, self.window_after))
            
            # Run parallel bootstrap fitting with lower priority
            with Pool(n_cores) as pool:
                # Use imap_unordered for better performance and progress tracking
                # chunksize=50 reduces overhead while maintaining responsiveness
                for task_result in pool.imap_unordered(_fit_single_bootstrap_replicate, tasks, chunksize=50):
                    # Extract results
                    results.append(task_result['result'])
                    if task_result['windowed'] is not None:
                        windowed_data_list.append(task_result['windowed'])
                    
                    fit_count += 1
                    
                    # Update progress bar
                    self.progress['value'] = fit_count
                    
                    # Update log every 10% or 1000 fits, whichever comes first
                    update_interval = min(1000, max(1, total_fits // 10))
                    if fit_count % update_interval == 0 or fit_count == total_fits:
                        percent = (fit_count / total_fits) * 100
                        self.log(f"    Progress: {fit_count}/{total_fits} ({percent:.1f}%)")
                        self.root.update()  # Force complete GUI update
        
        df_results = pd.DataFrame(results)
        df_windowed = pd.concat(windowed_data_list, ignore_index=True)
        
        # Ensure progress bar shows 100%
        self.progress['value'] = total_fits
        self.root.update_idletasks()
        
        self.log(f"\n  DEBUG: About to check behavioral context...")
        self.log(f"  DEBUG: df_results shape: {df_results.shape}")
        
        # Calculate behavioral context scores if OFF-food data is available
        if hasattr(self, 'off_food_results') and self.off_food_results is not None:
            try:
                self.log(f"  OFF-food results available: {len(self.off_food_results)} animals")
                self.log("  Calculating behavioral context scores for bootstrap replicates...")
                df_results = self.add_behavioral_context(df_results, "bootstrap")
                self.log("  Added food_detection_score and food_encounter_score columns")
            except Exception as e:
                self.log(f"  ERROR while calculating behavioral context: {str(e)}")
        else:
            self.log("  No OFF-food baselines available - skipping detection/encounter metrics")
        # Save CSV files
        df_results.to_csv(model_dir / "richards_fit_parameters.csv", index=False)
        df_windowed.to_csv(model_dir / "windowed_speed_data.csv", index=False)  # Bootstrap replicates (pooled samples)

        # Save run settings for downstream tools (e.g., permutation tests)
        try:
            settings = {
                "trajectory_aggregation": str(trajectory_agg or "median").strip().lower(),
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_sample_size": int(actual_sample_size),
                "fit_mode": "bootstrap"
            }
            with open(model_dir / "run_settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

        
        self.log(f"\n  Saved results to {model_dir}")
        self.log(f"  - richards_fit_parameters.csv: {len(df_results)} bootstrap replicate parameters")
        self.log(f"  - windowed_speed_data.csv: Pooled data from all bootstrap replicates")
        self.log(f"  - windowed_speed_data_original.csv: Original individual animal data for permutation testing")
        self.log(f"  Total replicates: {len(df_results)}")
        self.log(f"  Converged: {df_results['converged'].sum()} ({100*df_results['converged'].mean():.1f}%)")
        
        # Generate parameter summaries
        self.log(f"  Generating parameter group summaries...")
        self.generate_parameter_summaries(df_results, model_dir, "bootstrap")
        
        # Generate summary plots
        self.log(f"  Generating summary plots...")
        metrics = ["y_initial","y_final","A","B","M","nu"]
        
        # Add behavioral context scores if available
        if 'food_detection_score' in df_results.columns:
            metrics.extend(["food_detection_score", "food_encounter_score"])
        
        plot_count = 0
        for m in metrics:
            try:
                plot_summary_by_sex_strain_treatment(df_results, m, model_dir / f"{m}_by_sex_strain_treatment.png")
                plot_count += 1
            except Exception as e:
                self.log(f"    Warning: Could not generate plot for {m}: {str(e)}")
        
        self.log(f"  Generated {plot_count}/{len(metrics)} summary plots")
        
        # Generate R² distribution histogram
        self.plot_r2_distribution(df_results, model_dir, "bootstrap")
        
        # Generate aggregate curve plots
        self.log(f"  Generating aggregate curve comparison plots...")
        self.plot_aggregate_curves(df_results, df_windowed, model_dir)
        
        # Reset progress bar
        self.progress['value'] = 0
        self.root.update_idletasks()
        return df_results
    
    def add_behavioral_context(self, df_results, model_mode):
        """
        Add behavioral context scores to results (works for both bootstrap and individual modes).
        
        For each observation (bootstrap replicate or individual animal), calculate:
        - food_detection_score = y_initial / mean(OFF-food baseline for that GROUP)
        - food_encounter_score = y_final / mean(y_initial for that GROUP)
        
        BOTH modes use GROUP MEANS for denominators.
        The difference is what the group mean is calculated FROM:
        - Bootstrap mode: mean of 10,000 bootstrap OFF-food baselines
        - Individual mode: mean of 60 individual animal OFF-food baselines
        
        Args:
            df_results: Results DataFrame
            model_mode: 'bootstrap' or 'single'
        
        Returns:
            DataFrame with added behavioral context scores
        """
        df = df_results.copy()
        
        # Only use converged fits
        converged_mask = df['converged'] == True
        
        group_cols = ['treatment', 'sex', 'strain_genotype']
        
        # Initialize score columns
        df['food_detection_score'] = np.nan
        df['food_encounter_score'] = np.nan
        
        # Calculate OFF-food baselines (mean per group)
        # Both modes use the same logic here - just operating on different data
        off_food_baselines = self.off_food_results.groupby(group_cols)['baseline_speed'].mean().to_dict()
        
        # Calculate ON-food baselines (mean y_initial per group)
        on_food_baselines = df[converged_mask].groupby(group_cols)['y_initial'].mean().to_dict()
        
        # Calculate scores (same for both modes!)
        for idx, row in df[converged_mask].iterrows():
            group_key = (row['treatment'], row['sex'], row['strain_genotype'])
            
            # Food detection score: y_initial / mean(OFF-food baseline for GROUP)
            if group_key in off_food_baselines:
                off_baseline = off_food_baselines[group_key]
                df.at[idx, 'food_detection_score'] = row['y_initial'] / off_baseline
            
            # Food encounter score: y_final / mean(y_initial for GROUP)
            if group_key in on_food_baselines:
                on_baseline = on_food_baselines[group_key]
                df.at[idx, 'food_encounter_score'] = row['y_final'] / on_baseline
        
        return df
    
    def generate_parameter_summaries(self, df_results, model_dir, model_type):
        """
        Generate group-level summary statistics for Richards parameters.
        
        ROBUST STATISTICS:
        - Uses MEDIAN instead of mean (robust to outliers)
        - Uses PERCENTILES for confidence intervals (non-parametric)
        - Reports IQR instead of SD for spread
        """
        # Filter to converged fits only
        converged = df_results[df_results['converged'] == True].copy()
        
        if len(converged) == 0:
            self.log("    No converged fits for parameter summaries")
            return
        
        group_cols = ['treatment', 'sex', 'strain_genotype']
        
        # Determine which parameters to summarize based on model type
        if model_type in ("single", "bootstrap"):
            params_to_summarize = ['y_initial', 'y_final', 'A', 'B', 'M']
            # Add behavioral context scores if present
            if 'food_detection_score' in converged.columns:
                params_to_summarize.extend(['food_detection_score', 'food_encounter_score'])
        elif model_type == "double":
            params_to_summarize = ['y1_i', 'y1_f', 'y2_f', 'delta1', 'delta2', 
                                   'B1', 'M1', 'nu1', 'B2', 'M2', 'nu2']
        else:  # off_food_baseline
            params_to_summarize = ['y_initial', 'y_final', 'A', 'B', 'M']
        
        # Build summary for each group
        summaries = []
        for group_vals, group_data in converged.groupby(group_cols):
            treatment, sex, genotype = group_vals
            
            summary = {
                'treatment': treatment,
                'sex': sex,
                'genotype': genotype,
                'n': len(group_data)
            }
            
            # Calculate ROBUST statistics for each parameter
            for param in params_to_summarize:
                if param in group_data.columns:
                    values = group_data[param].dropna()
                    if len(values) > 0:
                        # ROBUST STATISTICS
                        summary[f'{param}_median'] = np.median(values)  # Central tendency
                        summary[f'{param}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)  # Spread
                        summary[f'{param}_mad'] = np.median(np.abs(values - np.median(values)))  # Robust SD
                        
                        # Percentile confidence intervals (always, not just for bootstrap)
                        summary[f'{param}_ci_lower'] = np.percentile(values, 2.5)
                        summary[f'{param}_ci_upper'] = np.percentile(values, 97.5)
                        
                        # Also keep mean/std for backward compatibility (but median is primary)
                        summary[f'{param}_mean'] = values.mean()
                        summary[f'{param}_std'] = values.std()
                    else:
                        summary[f'{param}_median'] = np.nan
                        summary[f'{param}_iqr'] = np.nan
                        summary[f'{param}_mad'] = np.nan
                        summary[f'{param}_ci_lower'] = np.nan
                        summary[f'{param}_ci_upper'] = np.nan
                        summary[f'{param}_mean'] = np.nan
                        summary[f'{param}_std'] = np.nan
            
            # Add R² statistics (also robust)
            r2_values = group_data['r_squared'].dropna()
            if len(r2_values) > 0:
                summary['r_squared_median'] = np.median(r2_values)
                summary['r_squared_iqr'] = np.percentile(r2_values, 75) - np.percentile(r2_values, 25)
                summary['r_squared_ci_lower'] = np.percentile(r2_values, 2.5)
                summary['r_squared_ci_upper'] = np.percentile(r2_values, 97.5)
                # Backward compatibility
                summary['r_squared_mean'] = r2_values.mean()
                summary['r_squared_std'] = r2_values.std()
            
            summaries.append(summary)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summaries)
        summary_path = model_dir / "parameter_group_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        self.log(f"    Saved ROBUST parameter summaries for {len(summary_df)} groups: {summary_path.name}")
        self.log(f"    (Using MEDIAN ± IQR and 95% percentile CIs)")
        
        # Log a preview of the summaries
        self.log(f"\n    Parameter Group Summaries ({model_type} model):")
        for _, row in summary_df.iterrows():
            self.log(f"      {row['treatment']} | {row['sex']} | {row.get('genotype', row.get('strain_genotype', ''))} (n={int(row['n'])})")

            # Show key parameters with ROBUST statistics (median + 95% percentile CI)
            if model_type in ("single", "bootstrap"):
                if 'y_initial_median' in row and not pd.isna(row['y_initial_median']):
                    self.log(
                        f"        y_initial: {row['y_initial_median']:.4f} "
                        f"[95% CI: {row['y_initial_ci_lower']:.4f}, {row['y_initial_ci_upper']:.4f}]"
                    )
                if 'y_final_median' in row and not pd.isna(row['y_final_median']):
                    self.log(
                        f"        y_final:   {row['y_final_median']:.4f} "
                        f"[95% CI: {row['y_final_ci_lower']:.4f}, {row['y_final_ci_upper']:.4f}]"
                    )
                if 'B_median' in row and not pd.isna(row['B_median']):
                    self.log(
                        f"        B:         {row['B_median']:.4f} "
                        f"[95% CI: {row['B_ci_lower']:.4f}, {row['B_ci_upper']:.4f}]"
                    )

                # Optional context scores
                if 'food_detection_score_median' in row and not pd.isna(row['food_detection_score_median']):
                    self.log(
                        f"        detection: {row['food_detection_score_median']:.4f} "
                        f"[95% CI: {row['food_detection_score_ci_lower']:.4f}, {row['food_detection_score_ci_upper']:.4f}]"
                    )
                if 'food_encounter_score_median' in row and not pd.isna(row['food_encounter_score_median']):
                    self.log(
                        f"        encounter: {row['food_encounter_score_median']:.4f} "
                        f"[95% CI: {row['food_encounter_score_ci_lower']:.4f}, {row['food_encounter_score_ci_upper']:.4f}]"
                    )

            elif model_type == "double":
                for p in ['y1_i','y1_f','y2_f','delta1','delta2','B1','M1','nu1','B2','M2','nu2']:
                    med_key = f"{p}_median"
                    lo_key = f"{p}_ci_lower"
                    hi_key = f"{p}_ci_upper"
                    if med_key in row and not pd.isna(row[med_key]):
                        self.log(
                            f"        {p}: {row[med_key]:.4f} "
                            f"[95% CI: {row[lo_key]:.4f}, {row[hi_key]:.4f}]"
                        )
    
    def plot_r2_distribution(self, df_results, model_dir, model_type):
        """Generate R² distribution histogram with optional threshold line"""
        converged = df_results[df_results["converged"] == True]
        if len(converged) == 0:
            return
        
        r2_values = converged['r_squared'].dropna()
        
        plt.figure(figsize=(8, 6))
        plt.hist(r2_values, bins=30, alpha=0.7, edgecolor='black')
        
        title_str = f'{model_type.title()} Model: R² Distribution'
        
        # Only add threshold line if R² flagging is enabled
        if self.enable_r2_flagging_var.get():
            r2_threshold = self.r2_threshold_var.get()
            plt.axvline(r2_threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold = {r2_threshold}')
            
            # Count and annotate
            n_below = (r2_values < r2_threshold).sum()
            n_above = (r2_values >= r2_threshold).sum()
            
            title_str += f'\nBelow threshold: {n_below} ({n_below/len(r2_values)*100:.1f}%) | ' \
                        f'Above threshold: {n_above} ({n_above/len(r2_values)*100:.1f}%)'
            plt.legend()
        else:
            title_str += f'\nTotal converged fits: {len(r2_values)}'
        
        plt.xlabel('R²')
        plt.ylabel('Count')
        plt.title(title_str)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_dir / "r2_distribution.png", bbox_inches="tight", dpi=150)
        plt.close()
    
    def plot_aggregate_curves(self, df_results, df_windowed, model_dir):
        """Generate aggregate curve comparison plots per group"""
        # Get ordered groups
        ordered_groups = create_hierarchical_groups(df_results)
        
        for treatment, sex, genotype in ordered_groups:
            # Get bootstrap replicates for this group
            group_mask = (
                (df_results['treatment'] == treatment) &
                (df_results['sex'] == sex) &
                (df_results['strain_genotype'] == genotype) &
                (df_results['converged'] == True)
            )
            group_results = df_results[group_mask]
            
            if len(group_results) == 0:
                continue
            
            # Get windowed data for this group
            data_mask = (
                (df_windowed['treatment'] == treatment) &
                (df_windowed['sex'] == sex) &
                (df_windowed['strain_genotype'] == genotype)
            )
            group_data = df_windowed[data_mask]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot 1: Raw mean trace with CI
            mean_trace = group_data.groupby('time_rel')['speed'].agg(['mean', 'sem']).reset_index()
            ci_lower = mean_trace['mean'] - 1.96 * mean_trace['sem']
            ci_upper = mean_trace['mean'] + 1.96 * mean_trace['sem']
            
            ax.plot(mean_trace['time_rel'], mean_trace['mean'], 'k-', 
                   linewidth=2, label='Raw mean trace', alpha=0.7)
            ax.fill_between(mean_trace['time_rel'], ci_lower, ci_upper, 
                           color='gray', alpha=0.2, label='Raw 95% CI')
            
            # Plot 2: Bootstrap aggregate curve with CI
            # Generate curves from all bootstrap replicates
            t_smooth = np.linspace(group_data['time_rel'].min(), group_data['time_rel'].max(), 500)
            bootstrap_curves = []
            
            for _, row in group_results.iterrows():
                y_curve = logistic_curve(t_smooth, row['y_initial'], row['y_final'], row['B'], row['M'])
                bootstrap_curves.append(y_curve)
            
            bootstrap_curves = np.array(bootstrap_curves)
            
            # Mean curve and CI from bootstrap distribution
            mean_curve = np.mean(bootstrap_curves, axis=0)
            ci_lower_boot = np.percentile(bootstrap_curves, 2.5, axis=0)
            ci_upper_boot = np.percentile(bootstrap_curves, 97.5, axis=0)
            
            ax.plot(t_smooth, mean_curve, 'b-', linewidth=2, 
                   label='Bootstrap mean curve')
            ax.fill_between(t_smooth, ci_lower_boot, ci_upper_boot, 
                           color='blue', alpha=0.2, label='Bootstrap 95% CI')
            
            # Formatting
            ax.axvline(0, color='red', linestyle='--', alpha=0.3, label='Encounter')
            ax.set_xlabel('Time Relative to Encounter (s)', fontsize=11)
            ax.set_ylabel('Speed (mm/s)', fontsize=11)
            ax.set_title(f'Aggregate Curve Comparison\n{genotype} | {sex} | {treatment}', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save with group name
            filename = f"aggregate_curve_{genotype}_{sex}_{treatment}.png"
            filename = filename.replace('|', '_').replace(' ', '_')
            plt.savefig(model_dir / filename, bbox_inches='tight', dpi=150)
            plt.close()
    
    def save_parameter_cheatsheet(self, model_dir, model_type):
        """Save parameter explanation cheat sheet"""
        cheat_path = model_dir / "PARAMETER_CHEATSHEET.txt"
        with open(cheat_path, "w") as f:
            f.write(f"Richards Curve Fitting - {model_type.upper()} Model\n")
            f.write("="*60 + "\n\n")
            if model_type == "single":
                f.write("Single Richards: y(t) = y_initial + A / (1 + exp(-B(t - M)))^(1/nu)\n\n")
                f.write("Parameters:\n")
                f.write("  y_initial: baseline speed before encounter\n")
                f.write("  y_final: asymptotic speed after encounter\n")
                f.write("  A = y_final - y_initial: total speed change\n")
                f.write("  B: steepness of transition (higher = faster change)\n")
                f.write("  M: midpoint time of transition (seconds)\n")
                f.write("  nu: asymmetry parameter (nu=1 => symmetric logistic)\n\n")
            else:
                f.write("Double Richards: two sequential transitions\n")
                f.write("  Phase 1: y1_i -> y1_f (B1, M1, nu1)\n")
                f.write("  Phase 2: y1_f -> y2_f (B2, M2, nu2)\n\n")
                f.write("Parameters:\n")
                f.write("  y_initial = y1_i: initial speed\n")
                f.write("  y_final = y2_f: final asymptotic speed\n")
                f.write("  A = y2_f - y1_i: total speed change\n")
                f.write("  delta1 = y1_f - y1_i: phase 1 change\n")
                f.write("  delta2 = y2_f - y1_f: phase 2 change\n")
                f.write("  B1, M1, nu1: phase 1 steepness, midpoint, asymmetry\n")
                f.write("  B2, M2, nu2: phase 2 steepness, midpoint, asymmetry\n\n")
            f.write("Fit Quality:\n")
            f.write("  R²: proportion of variance explained (0–1, higher is better)\n")
            f.write("  RMSE: root mean squared error (lower is better)\n")
            f.write("  AIC: Akaike Information Criterion (lower is better)\n")
            f.write("  BIC: Bayesian Information Criterion (lower is better, penalizes complexity)\n")
    
    def report_model_summary(self, results_df, model):
        converged = results_df[results_df["converged"] == True]
        self.log(f"\n{model.upper()} MODEL SUMMARY:")
        self.log(f"  Total animals: {len(results_df)}")
        self.log(f"  Converged fits: {len(converged)} ({len(converged)/len(results_df)*100:.1f}%)")
        if len(converged) > 0:
            self.log(f"  Mean R²: {converged['r_squared'].mean():.3f}")
            self.log(f"  Median R²: {converged['r_squared'].median():.3f}")
            
            # R² threshold analysis - only if enabled
            if self.enable_r2_flagging_var.get():
                r2_threshold = self.r2_threshold_var.get()
                poor_fits = converged[converged['r_squared'] < r2_threshold]
                n_poor = len(poor_fits)
                pct_poor = (n_poor / len(converged) * 100) if len(converged) > 0 else 0
                
                self.log(f"\n  R² THRESHOLD ANALYSIS (threshold = {r2_threshold}):")
                self.log(f"  Fits below threshold: {n_poor} ({pct_poor:.1f}%)")
                self.log(f"  Fits above threshold: {len(converged) - n_poor} ({100-pct_poor:.1f}%)")
                
                if n_poor > 0:
                    self.log(f"  Mean R² of poor fits: {poor_fits['r_squared'].mean():.3f}")
                    self.log(f"  Range of poor fits: {poor_fits['r_squared'].min():.3f} - {poor_fits['r_squared'].max():.3f}")
                    
                    # Save poor fits report
                    model_dir = self.output_dir / f"model_{model}"
                    poor_fits_path = model_dir / f"poor_fits_below_r2_{r2_threshold}.csv"
                    poor_fits.to_csv(poor_fits_path, index=False)
                    self.log(f"  Saved poor fits report: {poor_fits_path.name}")
    
    def compare_models(self):
        """Compare single vs double models and save comparison files"""
        # Clean up old comparison files
        comparison_files = [
            "model_comparison_summary.csv",
            "r2_scatter_single_vs_double.png",
            "r2_violin_single_vs_double.png",
            "delta_AIC_hist.png",
            "delta_BIC_hist.png"
        ]
        for fname in comparison_files:
            fpath = self.output_dir / fname
            if fpath.exists():
                fpath.unlink()
        
        m = pd.merge(
            self.results_single[["assay_num","track_num","r_squared","aic","bic"]],
            self.results_double[["assay_num","track_num","r_squared","aic","bic"]],
            on=["assay_num","track_num"], 
            suffixes=("_single","_double"), 
            how="inner"
        )
        
        if m.empty:
            self.log("No overlapping animals for comparison")
            return
        
        m["ΔR²"] = m["r_squared_double"] - m["r_squared_single"]
        m["ΔAIC"] = m["aic_single"] - m["aic_double"]
        m["ΔBIC"] = m["bic_single"] - m["bic_double"]
        
        # Add preferred model based on BIC
        def pref_from_delta_bic(delta_bic):
            if pd.isna(delta_bic): return "Unknown"
            if delta_bic > 10: return "Double"
            if delta_bic < -10: return "Single"
            return "Similar"
        m["preferred_model"] = m["ΔBIC"].apply(pref_from_delta_bic)
        
        # Save comparison summary
        summary_path = self.output_dir / "model_comparison_summary.csv"
        m.to_csv(summary_path, index=False)
        self.log(f"  Saved comparison summary: {summary_path.name}")
        
        # Annotate individual model CSVs with comparison data
        single_path = self.output_dir / "model_single" / "richards_fit_parameters.csv"
        double_path = self.output_dir / "model_double" / "richards_fit_parameters.csv"
        
        if single_path.exists():
            ds = pd.read_csv(single_path)
            ds_aug = pd.merge(ds, m[["assay_num","track_num","r_squared_double","aic_double","bic_double","ΔR²","ΔAIC","ΔBIC","preferred_model"]],
                            on=["assay_num","track_num"], how="left")
            ds_aug.rename(columns={"r_squared_double":"r_squared_other","aic_double":"aic_other","bic_double":"bic_other",
                                  "ΔR²":"delta_r2","ΔAIC":"delta_aic","ΔBIC":"delta_bic"}, inplace=True)
            ds_aug.to_csv(single_path, index=False)
        
        if double_path.exists():
            dd = pd.read_csv(double_path)
            dd_aug = pd.merge(dd, m[["assay_num","track_num","r_squared_single","aic_single","bic_single","ΔR²","ΔAIC","ΔBIC","preferred_model"]],
                            on=["assay_num","track_num"], how="left")
            dd_aug.rename(columns={"r_squared_single":"r_squared_other","aic_single":"aic_other","bic_single":"bic_other",
                                  "ΔR²":"delta_r2","ΔAIC":"delta_aic","ΔBIC":"delta_bic"}, inplace=True)
            dd_aug.to_csv(double_path, index=False)
        
        # Generate comparison plots
        self.log(f"  Generating comparison plots...")
        self.save_comparison_plots(m)
        
        # Log statistics
        self.log(f"  Compared {len(m)} animals")
        self.log(f"  Mean ΔR² (Double - Single): {m['ΔR²'].mean():.4f}")
        self.log(f"  Mean ΔBIC (Single - Double): {m['ΔBIC'].mean():.2f}")
        
        # Preference based on BIC
        double_better = (m["ΔBIC"] > 10).sum()
        single_better = (m["ΔBIC"] < -10).sum()
        similar = len(m) - double_better - single_better
        
        self.log(f"\nModel Preference (by BIC):")
        self.log(f"  Double better: {double_better} ({double_better/len(m)*100:.1f}%)")
        self.log(f"  Single better: {single_better} ({single_better/len(m)*100:.1f}%)")
        self.log(f"  Similar: {similar} ({similar/len(m)*100:.1f}%)")
    
    def save_comparison_plots(self, m):
        """Generate and save model comparison plots"""
        # R² scatter
        self.root.update_idletasks()  # Update GUI
        plt.figure(figsize=(6,6))
        plt.scatter(m["r_squared_single"], m["r_squared_double"], alpha=0.55, edgecolors="black", linewidths=0.4)
        plt.plot([0,1],[0,1], "--", color="gray")
        plt.xlim(0,1); plt.ylim(0,1)
        plt.xlabel("R² (Single)"); plt.ylabel("R² (Double)")
        plt.title("R² Comparison Per Animal")
        plt.tight_layout()
        plt.savefig(self.output_dir / "r2_scatter_single_vs_double.png", bbox_inches="tight", dpi=150)
        plt.close()
        
        # R² violin
        self.root.update_idletasks()  # Update GUI
        plt.figure(figsize=(6,5))
        r2s = m["r_squared_single"].dropna().values
        r2d = m["r_squared_double"].dropna().values
        parts = plt.violinplot([r2s, r2d], positions=[1,2], showmeans=True, showmedians=True)
        for body in parts["bodies"]:
            body.set_alpha(0.75)
        plt.xticks([1,2], ["Single","Double"])
        plt.ylabel("R²"); plt.title("R² Distribution by Model")
        plt.tight_layout()
        plt.savefig(self.output_dir / "r2_violin_single_vs_double.png", bbox_inches="tight", dpi=150)
        plt.close()
        
        # ΔAIC / ΔBIC histograms
        for col, fname in [("ΔAIC","delta_AIC_hist.png"), ("ΔBIC","delta_BIC_hist.png")]:
            self.root.update_idletasks()  # Update GUI
            plt.figure(figsize=(7,5))
            vals = m[col].dropna().values
            if len(vals): 
                plt.hist(vals, bins=30)
            plt.axvline(0, color="black", linestyle="--")
            plt.xlabel(f"{col} (Single − Double)")
            plt.ylabel("Count")
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(self.output_dir / fname, bbox_inches="tight", dpi=150)
            plt.close()
    
    def calculate_behavioral_context(self):
        """Calculate food detection and food encounter behavioral contexts"""
        # OFF-food is optional; only compute detection/encounter metrics if OFF-food baselines exist
        if not (hasattr(self, 'off_food_results') and self.off_food_results is not None and len(self.off_food_results) > 0):
            self.log("\nSkipping behavioral context analysis (no OFF-food baseline loaded)")
            return
        mode = 'both'
        
        self.log("\n" + "="*70)
        self.log("BEHAVIORAL CONTEXT ANALYSIS")
        self.log("="*70)
        
        # Check if we're using bootstrap mode
        model_choice = self.model_var.get()
        is_bootstrap = (model_choice == "bootstrap")
        
        # Determine which model to use
        if self.results_single is not None:
            on_food_results = self.results_single.copy()
            model_used = "bootstrap" if is_bootstrap else "single"
        else:
            self.log("ERROR: No model results available for behavioral context analysis")
            return
        
        self.log(f"Using {model_used} model results")
        
        if is_bootstrap:
            self.log(f"Bootstrap replicates: {len(on_food_results)}")
        else:
            self.log(f"ON-food animals: {len(on_food_results)}")
        
        # Filter to converged fits only
        on_food_results = on_food_results[on_food_results['converged'] == True].copy()
        self.log(f"Converged fits: {len(on_food_results)}")
        
        if len(on_food_results) == 0:
            self.log("ERROR: No converged fits for behavioral context analysis")
            return
        
        # Grouping columns
        group_cols = ['treatment', 'sex', 'strain_genotype']
        
        if mode == "both":
            self.log("\n" + "-"*60)
            self.log("FOOD DETECTION CONTEXT (OFF-food baseline)")
            self.log("-"*60)
            
            # Use OFF-food results that were calculated as direct mean speeds
            if not hasattr(self, 'off_food_results') or self.off_food_results is None:
                self.log("ERROR: OFF-food results not available")
                return
            
            off_food_results = self.off_food_results.copy()
            
            self.log(f"OFF-food baseline animals: {len(off_food_results)}")
            
            # Check if OFF-food has treatment variation
            off_treatments = off_food_results['treatment'].unique()
            self.log(f"OFF-food treatments: {list(off_treatments)}")
            
            # Determine grouping strategy
            if len(off_treatments) == 1:
                # OFF-food has no treatment variation - group by sex and genotype ONLY
                self.log("OFF-food has single treatment - using sex × genotype grouping")
                off_group_cols = ['sex', 'strain_genotype']
                
                off_baselines = off_food_results.groupby(off_group_cols)['baseline_speed'].agg(['mean', 'std', 'count']).reset_index()
                off_baselines.columns = ['sex', 'strain_genotype', 'off_food_baseline', 'off_food_std', 'off_food_n']
                
                self.log("\nOFF-food baselines by group:")
                for _, row in off_baselines.iterrows():
                    self.log(f"  {row['sex']} | {row['strain_genotype']}: " +
                            f"{row['off_food_baseline']:.4f} ± {row['off_food_std']:.4f} mm/s (n={int(row['off_food_n'])})")
                
                # Merge baselines with ON-food data (baseline applies to all treatments)
                on_food_results = on_food_results.merge(
                    off_baselines[['sex', 'strain_genotype', 'off_food_baseline']], 
                    on=['sex', 'strain_genotype'], 
                    how='left'
                )
            else:
                # OFF-food has treatment variation - group by treatment × sex × genotype
                self.log("OFF-food has multiple treatments - using treatment × sex × genotype grouping")
                
                off_baselines = off_food_results.groupby(group_cols)['baseline_speed'].agg(['mean', 'std', 'count']).reset_index()
                off_baselines.columns = ['treatment', 'sex', 'strain_genotype', 'off_food_baseline', 'off_food_std', 'off_food_n']
                
                self.log("\nOFF-food baselines by group:")
                for _, row in off_baselines.iterrows():
                    self.log(f"  {row['treatment']} | {row['sex']} | {row['strain_genotype']}: " +
                            f"{row['off_food_baseline']:.4f} ± {row['off_food_std']:.4f} mm/s (n={int(row['off_food_n'])})")
                
                # Merge baselines with ON-food data
                on_food_results = on_food_results.merge(
                    off_baselines[['treatment', 'sex', 'strain_genotype', 'off_food_baseline']], 
                    on=group_cols, 
                    how='left'
                )
            
            # Check if scores are already calculated (from add_behavioral_context)
            if 'food_detection_score' in on_food_results.columns and on_food_results['food_detection_score'].notna().any():
                self.log("\nFood detection scores already calculated")
                self.log(f"  Using pre-calculated scores from {model_used} mode")
            else:
                # Calculate food detection score (fallback - should not normally happen)
                self.log("\nCalculating food detection scores...")
                on_food_results['food_detection_score'] = on_food_results['y_initial'] / on_food_results['off_food_baseline']
            
            self.log(f"Food detection scores available for {on_food_results['food_detection_score'].notna().sum()} animals")
            
            # Check for missing baselines
            missing_baselines = on_food_results['off_food_baseline'].isna().sum()
            if missing_baselines > 0:
                self.log(f"  WARNING: {missing_baselines} animals missing OFF-food baseline (no matching group)")

        
        # Food encounter context (always calculated for on_food_only and both modes)
        self.log("\n" + "-"*60)
        self.log("FOOD ENCOUNTER CONTEXT (ON-food before baseline)")
        self.log("-"*60)
        
        # Calculate mean ON-food y_initial per group (for summary display)
        on_baselines = on_food_results.groupby(group_cols)['y_initial'].agg(['mean', 'std', 'count']).reset_index()
        on_baselines.columns = ['treatment', 'sex', 'strain_genotype', 'on_food_baseline', 'on_food_std', 'on_food_n']
        
        self.log("\nON-food before baselines by group:")
        for _, row in on_baselines.iterrows():
            self.log(f"  {row['treatment']} | {row['sex']} | {row['strain_genotype']}: " +
                    f"{row['on_food_baseline']:.4f} ± {row['on_food_std']:.4f} mm/s (n={int(row['on_food_n'])})")
        
        # Check if food_encounter_score already calculated
        if 'food_encounter_score' in on_food_results.columns and on_food_results['food_encounter_score'].notna().any():
            self.log("\nFood encounter scores already calculated")
            self.log(f"  Using pre-calculated scores from {model_used} mode")
        else:
            # Calculate food encounter score (fallback - should not normally happen)
            self.log("\nCalculating food encounter scores...")
            # Merge baselines
            on_food_results = on_food_results.merge(
                on_baselines[['treatment', 'sex', 'strain_genotype', 'on_food_baseline']], 
                on=group_cols, 
                how='left'
            )
            on_food_results['food_encounter_score'] = on_food_results['y_final'] / on_food_results['on_food_baseline']
        
        self.log(f"Food encounter scores available for {on_food_results['food_encounter_score'].notna().sum()} animals")
        
        # Store results
        self.behavioral_context_results = on_food_results
        
        # Calculate group summaries
        self.calculate_behavioral_context_summaries()
        
        # Save results
        self.save_behavioral_context_results()
        
        # Generate plots
        self.plot_behavioral_context()
        
        self.log("\n" + "="*70)
        self.log("BEHAVIORAL CONTEXT ANALYSIS COMPLETE")
        self.log("="*70)
    
    def calculate_behavioral_context_summaries(self):
        """Calculate group-level summary statistics for behavioral contexts"""
        group_cols = ['treatment', 'sex', 'strain_genotype']
        
        summaries = []
        for group_vals, group_data in self.behavioral_context_results.groupby(group_cols):
            treatment, sex, genotype = group_vals
            
            summary = {
                'treatment': treatment,
                'sex': sex,
                'genotype': genotype,
                'n': len(group_data)
            }
            
            # y_initial and y_final statistics
            summary['y_initial_mean'] = group_data['y_initial'].mean()
            summary['y_initial_std'] = group_data['y_initial'].std()
            summary['y_initial_sem'] = group_data['y_initial'].sem()
            
            summary['y_final_mean'] = group_data['y_final'].mean()
            summary['y_final_std'] = group_data['y_final'].std()
            summary['y_final_sem'] = group_data['y_final'].sem()
            
            # Food detection scores (if available)
            if 'food_detection_score' in group_data.columns:
                detection_scores = group_data['food_detection_score'].dropna()
                if len(detection_scores) > 0:
                    summary['food_detection_mean'] = detection_scores.mean()
                    summary['food_detection_std'] = detection_scores.std()
                    summary['food_detection_sem'] = detection_scores.sem()
                else:
                    summary['food_detection_mean'] = np.nan
                    summary['food_detection_std'] = np.nan
                    summary['food_detection_sem'] = np.nan
            
            # Food encounter scores
            encounter_scores = group_data['food_encounter_score'].dropna()
            if len(encounter_scores) > 0:
                summary['food_encounter_mean'] = encounter_scores.mean()
                summary['food_encounter_std'] = encounter_scores.std()
                summary['food_encounter_sem'] = encounter_scores.sem()
            else:
                summary['food_encounter_mean'] = np.nan
                summary['food_encounter_std'] = np.nan
                summary['food_encounter_sem'] = np.nan
            
            summaries.append(summary)
        
        self.behavioral_context_summary = pd.DataFrame(summaries)
        self.log(f"\nCalculated summaries for {len(self.behavioral_context_summary)} groups")
    
    def save_behavioral_context_results(self):
        """Save behavioral context results to CSV files"""
        context_dir = self.output_dir / "behavioral_context"
        context_dir.mkdir(exist_ok=True)
        
        # Save per-animal results
        animal_path = context_dir / "behavioral_context_per_animal.csv"
        self.behavioral_context_results.to_csv(animal_path, index=False)
        self.log(f"\nSaved per-animal context scores: {animal_path.name}")
        
        # Save group summaries
        summary_path = context_dir / "behavioral_context_group_summary.csv"
        self.behavioral_context_summary.to_csv(summary_path, index=False)
        self.log(f"Saved group summary: {summary_path.name}")
    
    def plot_behavioral_context(self):
        """Generate plots for behavioral context analysis"""
        context_dir = self.output_dir / "behavioral_context"
        
        mode = self.context_mode.get()
        
        # Get hierarchically ordered groups from the results data
        ordered_group_tuples = create_hierarchical_groups(self.behavioral_context_results)
        
        # Create ordered labels
        group_labels = []
        for treatment, sex, genotype in ordered_group_tuples:
            group_labels.append(format_group_label(treatment, sex, genotype))
        
        if not group_labels:
            self.log("  No groups to plot")
            return
        
        self.log("\nGenerating behavioral context plots...")
        
        # Plot 1: Food encounter scores by group (VIOLIN PLOT)
        self.root.update_idletasks()
        fig, ax = plt.subplots(figsize=(max(8, len(group_labels)*0.8), 6))
        
        # Prepare data for violin plot
        encounter_data = []
        for treatment, sex, genotype in ordered_group_tuples:
            mask = (
                (self.behavioral_context_results['treatment'] == treatment) &
                (self.behavioral_context_results['sex'] == sex) &
                (self.behavioral_context_results['strain_genotype'] == genotype)
            )
            group_scores = self.behavioral_context_results[mask]['food_encounter_score'].dropna()
            encounter_data.append(group_scores.values)
        
        x_pos = list(range(1, len(group_labels) + 1))
        parts = ax.violinplot(encounter_data, positions=x_pos, showmeans=True, showmedians=True)
        
        # Style violin plots
        for body in parts["bodies"]:
            body.set_alpha(0.7)
        
        ax.set_xlabel('Group (Genotype|Sex|Treatment)', fontsize=11)
        ax.set_ylabel('Food Encounter Score\n(y_final / y_initial baseline)', fontsize=11)
        ax.set_title('Food Encounter Context by Group', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='No change')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(context_dir / "food_encounter_by_group.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Food detection scores by group (VIOLIN PLOT, if available)
        if mode == "both" and 'food_detection_score' in self.behavioral_context_results.columns:
            self.root.update_idletasks()
            fig, ax = plt.subplots(figsize=(max(8, len(group_labels)*0.8), 6))
            
            # Prepare data for violin plot
            detection_data = []
            for treatment, sex, genotype in ordered_group_tuples:
                mask = (
                    (self.behavioral_context_results['treatment'] == treatment) &
                    (self.behavioral_context_results['sex'] == sex) &
                    (self.behavioral_context_results['strain_genotype'] == genotype)
                )
                group_scores = self.behavioral_context_results[mask]['food_detection_score'].dropna()
                detection_data.append(group_scores.values)
            
            x_pos = list(range(1, len(group_labels) + 1))
            parts = ax.violinplot(detection_data, positions=x_pos, showmeans=True, showmedians=True)
            
            # Style violin plots
            for body in parts["bodies"]:
                body.set_alpha(0.7)
                body.set_facecolor('orange')
                body.set_edgecolor('black')
            
            ax.set_xlabel('Group (Genotype|Sex|Treatment)', fontsize=11)
            ax.set_ylabel('Food Detection Score\n(y_initial ON / y_initial OFF)', fontsize=11)
            ax.set_title('Food Detection Context by Group', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_labels, rotation=45, ha='right')
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='No detection')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(context_dir / "food_detection_by_group.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Detection vs Encounter scatter (if both available)
        if mode == "both" and 'food_detection_score' in self.behavioral_context_results.columns:
            self.root.update_idletasks()
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Color by genotype (hierarchical coloring)
            genotypes = sort_genotypes(self.behavioral_context_results['strain_genotype'].unique())
            colors = plt.cm.Set2(np.linspace(0, 1, len(genotypes)))
            
            for genotype, color in zip(genotypes, colors):
                subset = self.behavioral_context_results[self.behavioral_context_results['strain_genotype'] == genotype]
                ax.scatter(subset['food_detection_score'], subset['food_encounter_score'],
                          label=genotype, alpha=0.6, s=50, color=color, edgecolors='black', linewidths=0.5)
            
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Food Detection Score\n(y_initial ON / y_initial OFF)', fontsize=11)
            ax.set_ylabel('Food Encounter Score\n(y_final / y_initial ON)', fontsize=11)
            ax.set_title('Food Detection vs Food Encounter', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(context_dir / "detection_vs_encounter_scatter.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        self.log(f"Saved {3 if mode == 'both' else 2} behavioral context plots")
    
    def save_results(self):
        """Save preprocessing summary (main results already saved during analysis)"""
        if self.output_dir is None:
            messagebox.showwarning("Warning", "No output directory selected")
            return
        
        if self.results_single is None and self.results_double is None:
            messagebox.showwarning("Warning", "No results to save. Run analysis first.")
            return
        
        try:
            # Save outlier report if filtering was used
            if self.outlier_report is not None and not self.outlier_report.empty:
                outlier_path = self.output_dir / "outlier_filtering_report.csv"
                self.outlier_report.to_csv(outlier_path, index=False)
                self.log(f"Saved outlier report: {outlier_path}")
            
            # Save preprocessing summary
            n_removed = 0
            pct_removed = 0
            if self.outlier_report is not None and not self.outlier_report.empty:
                n_removed = self.outlier_report['n_outliers'].sum()
                n_total = self.outlier_report['n_points'].sum()
                pct_removed = (n_removed / n_total * 100) if n_total > 0 else 0
            
            preproc_summary = {
                "filtering_enabled": self.enable_filtering_var.get(),
                "smoothing_enabled": self.enable_smoothing_var.get(),
                "pixels_per_mm": self.pixels_per_mm,
                "window_before": self.window_before,
                "window_after": self.window_after,
            }
            if self.enable_filtering_var.get():
                preproc_summary.update({
                    "percentile_lower": self.percentile_lower,
                    "percentile_upper": self.percentile_upper,
                    "mad_threshold": self.mad_threshold,
                    "points_removed": n_removed,
                    "pct_removed": pct_removed,
                })
            if self.enable_smoothing_var.get():
                preproc_summary.update({
                    "smooth_method": self.smooth_method_var.get(),
                    "smooth_window": self.smooth_window,
                })
            
            pd.DataFrame([preproc_summary]).to_csv(self.output_dir / "preprocessing_summary.csv", index=False)
            self.log(f"Saved preprocessing summary: preprocessing_summary.csv")
            
            messagebox.showinfo("Success", f"Additional files saved to:\n{self.output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save files:\n{str(e)}")
            self.log(f"ERROR saving files: {str(e)}")


    def save_permutation_package(self):
        """Save permutation-ready package combining ON-food and OFF-food data."""
        if not hasattr(self, 'results_single') or self.results_single is None:
            messagebox.showwarning("Warning", 
                "No bootstrap results available.\n"
                "Run bootstrap analysis first.")
            return
        
        try:
            import json
            import shutil
            
            perm_dir = self.output_dir / "permutation_data"
            perm_dir.mkdir(exist_ok=True)
            
            self.log("\n" + "="*70)
            self.log("SAVING PERMUTATION PACKAGE")
            self.log("="*70)
            
            # 1. Combined bootstrap results (ON-food + OFF-food)
            self.log("\n1. Creating combined bootstrap results...")
            bootstrap_combined = []
            
            # Add ON-food bootstrap results
            param_cols = ['y_initial', 'y_final', 'A', 'B', 'M']
            
            # Add behavioral context scores if they exist
            if 'food_detection_score' in self.results_single.columns:
                param_cols.extend(['food_detection_score', 'food_encounter_score'])
                self.log("  Including behavioral context scores (food_detection_score, food_encounter_score)")
            
            for _, row in self.results_single.iterrows():
                for param in param_cols:
                    if param in self.results_single.columns and pd.notna(row[param]):
                        bootstrap_combined.append({
                            'data_type': 'on_food',
                            'bootstrap_replicate': int(row['bootstrap_replicate']),
                            'treatment': str(row['treatment']),
                            'sex': str(row['sex']),
                            'strain_genotype': str(row['strain_genotype']),
                            'parameter': param,
                            'value': float(row[param])
                        })
            
            n_on = len(bootstrap_combined)
            self.log(f"  Added {n_on} ON-food bootstrap values")
            
            # Add OFF-food bootstrap results if available
            if hasattr(self, 'off_food_results') and self.off_food_results is not None:
                if 'bootstrap_replicate' in self.off_food_results.columns:
                    for _, row in self.off_food_results.iterrows():
                        if pd.notna(row['baseline_speed']):
                            bootstrap_combined.append({
                                'data_type': 'off_food',
                                'bootstrap_replicate': int(row['bootstrap_replicate']),
                                'treatment': str(row['treatment']),
                                'sex': str(row['sex']),
                                'strain_genotype': str(row['strain_genotype']),
                                'parameter': 'baseline_speed',
                                'value': float(row['baseline_speed'])
                            })
                    
                    n_off = len(bootstrap_combined) - n_on
                    self.log(f"  Added {n_off} OFF-food bootstrap values")
                else:
                    self.log("  OFF-food data in individual mode (no bootstrap)")
            else:
                self.log("  No OFF-food data available")
            
            # Save combined bootstrap
            df_bootstrap = pd.DataFrame(bootstrap_combined)
            bootstrap_path = perm_dir / "bootstrap_results_combined.csv"
            df_bootstrap.to_csv(bootstrap_path, index=False)
            self.log(f"  ✓ Saved: {bootstrap_path.name} ({len(df_bootstrap)} rows)")
            
            # 2. Copy ON-food windowed data
            self.log("\n2. Copying ON-food windowed data...")
            # Construct path to windowed data (saved during bootstrap analysis)
            windowed_source = self.output_dir / "bootstrap_model" / "windowed_speed_data_original.csv"
            if windowed_source.exists():
                windowed_dest = perm_dir / "on_food_windowed.csv"
                shutil.copy(windowed_source, windowed_dest)
                self.log(f"  ✓ Saved: {windowed_dest.name}")
            else:
                # Try single_model directory as fallback
                windowed_source = self.output_dir / "single_model" / "windowed_speed_data_original.csv"
                if windowed_source.exists():
                    windowed_dest = perm_dir / "on_food_windowed.csv"
                    shutil.copy(windowed_source, windowed_dest)
                    self.log(f"  ✓ Saved: {windowed_dest.name}")
                else:
                    self.log("  ✗ Warning: Windowed data file not found")
            
            # 3. Copy OFF-food individual baselines
            self.log("\n3. Copying OFF-food individual baselines...")
            off_individual_path = self.output_dir / "off_food_baselines" / "off_food_individual_baselines.csv"
            if off_individual_path.exists():
                off_dest = perm_dir / "off_food_individual_baselines.csv"
                shutil.copy(off_individual_path, off_dest)
                off_df = pd.read_csv(off_individual_path)
                self.log(f"  ✓ Saved: {off_dest.name} ({len(off_df)} animals)")
            else:
                self.log("  ℹ No OFF-food individual baselines (OFF-food not loaded)")
            
            # 4. Save metadata
            self.log("\n4. Saving metadata...")
            # Determine which windowed file was used
            windowed_file_used = None
            if (self.output_dir / "bootstrap_model" / "windowed_speed_data_original.csv").exists():
                windowed_file_used = str(self.output_dir / "bootstrap_model" / "windowed_speed_data_original.csv")
            elif (self.output_dir / "single_model" / "windowed_speed_data_original.csv").exists():
                windowed_file_used = str(self.output_dir / "single_model" / "windowed_speed_data_original.csv")
            
            metadata = {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'n_bootstrap': int(self.n_bootstrap_var.get()),
                'group_columns': ['treatment', 'sex', 'strain_genotype'],
                'on_food_parameters': param_cols,
                'off_food_parameters': ['baseline_speed'],
                'source_files': {
                    'windowed': windowed_file_used,
                    'off_food': str(self.off_food_path) if hasattr(self, 'off_food_path') and self.off_food_path else None
                }
            }
            
            metadata_path = perm_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.log(f"  ✓ Saved: {metadata_path.name}")
            
            # Summary
            self.log("\n" + "="*70)
            self.log("✓ PERMUTATION PACKAGE SAVED SUCCESSFULLY!")
            self.log("="*70)
            self.log(f"\nLocation: {perm_dir}")
            self.log("\nFiles created:")
            self.log("  ✓ bootstrap_results_combined.csv")
            self.log("  ✓ on_food_windowed.csv")
            if off_individual_path.exists():
                self.log("  ✓ off_food_individual_baselines.csv")
            self.log("  ✓ metadata.json")
            self.log("\nLoad this directory in the permutation testing app.")
            
            messagebox.showinfo(
                "Success",
                f"Permutation package saved to:\n{perm_dir}\n\n"
                f"Load this directory in the permutation testing app."
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log(f"\n✗ ERROR saving permutation package:\n{error_msg}")
            messagebox.showerror("Error", f"Failed to save package:\n{str(e)}")


def main():
    root = tk.Tk()
    app = RichardsCurveFitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()