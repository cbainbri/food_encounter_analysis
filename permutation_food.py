#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Permutation Testing Application - Self-Contained [BUGFIX VERSION]
Double-click this file to launch the GUI

Post-hoc analysis for Richards curve fitting with bootstrap bagging.
Provides both Standard and Max-T permutation tests.

BUGFIX: This version properly updates the comparison dropdowns when loading
a permutation package. The original version loaded the data but forgot to
populate the dropdown menus, making them appear empty.
"""

import tkinter as tk
import json
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns  # (imported but not required; left for backward compatibility)
from dataclasses import dataclass
import warnings
from tqdm import tqdm  # (imported but not required; left for backward compatibility)
import sys
import io
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Comparison:
    """Represents a single pairwise comparison."""
    name: str
    group_a: str
    group_b: str
    observed_diff: float
    group_a_mean: float
    group_b_mean: float


@dataclass
class PermutationResult:
    """Results from permutation test."""
    comparison_name: str
    observed_diff: float
    p_value: float
    null_distribution: np.ndarray
    method: str
    n_permutations: int


@dataclass
class OneSampleResult:
    """Result from a one-sample bootstrap CI test vs a reference value."""
    group: str
    parameter: str
    mu0: float
    mean: float
    ci_lo: float       # 2.5th percentile of bootstrap distribution
    ci_hi: float       # 97.5th percentile of bootstrap distribution
    mean_minus_mu0: float
    p_value: float     # two-tailed empirical p: how far mu0 sits in the tail
    significant: bool  # primary readout: mu0 outside 95% CI
    n_bootstrap: int


# =============================================================================
# Parallel Processing Worker Functions
# =============================================================================

def _permutation_worker_standard(args):
    """
    Worker function for CLASSICAL permutation test (no bootstrap within permutation).
    Must be at module level for multiprocessing.

    Denominators for score parameters are fixed per group (pre-computed before the
    permutation loop) and passed in explicitly. This ensures the null distribution
    is generated with the same denominator convention as the observed statistic,
    preventing artificial inflation of null variance.

    Args:
        args: (perm_idx, all_animals_copy, n_a, parameter, trajectory_agg,
               denom_a, denom_b)
            denom_a: Pre-computed denominator for group A (scalar, or np.nan if
                     parameter needs no denominator).
            denom_b: Pre-computed denominator for group B (same).

    Returns:
        null_diff or None if fit failed
    """
    perm_idx, all_animals_copy, n_a, parameter, trajectory_agg, denom_a, denom_b = args

    # Shuffle animals WITHOUT replacement (classical permutation)
    np.random.shuffle(all_animals_copy)

    # Split into two groups of original sizes
    perm_a = all_animals_copy[:n_a]
    perm_b = all_animals_copy[n_a:]

    # Aggregate ALL animals and fit (NO sampling/bootstrap).
    # Pass fixed denominators so the score is always computed as
    #   numerator(shuffled animals) / fixed_group_denominator
    # rather than looking up the denominator from whichever animal happens
    # to appear first in the shuffled subgroup.
    est_a = aggregate_all_and_fit(perm_a, parameter, trajectory_agg,
                                  explicit_denominator=denom_a)
    est_b = aggregate_all_and_fit(perm_b, parameter, trajectory_agg,
                                  explicit_denominator=denom_b)

    if not np.isnan(est_a) and not np.isnan(est_b):
        return est_a - est_b
    else:
        return None


def _permutation_worker_maxt(args):
    """
    Worker function for CLASSICAL Max-T permutation test (no bootstrap within permutation).
    Must be at module level for multiprocessing.

    Denominators are fixed per group (see _permutation_worker_standard for rationale).

    Args:
        args: (perm_idx, comparisons_data, parameter, trajectory_agg)
        comparisons_data: List of (group_a_animals, group_b_animals, denom_a, denom_b)

    Returns:
        max_stat or None if all fits failed
    """
    perm_idx, comparisons_data, parameter, trajectory_agg = args

    perm_stats = []

    for group_a_animals, group_b_animals, denom_a, denom_b in comparisons_data:
        # Pool and shuffle WITHOUT replacement
        all_animals = group_a_animals + group_b_animals
        np.random.shuffle(all_animals)

        n_a = len(group_a_animals)
        perm_a = all_animals[:n_a]
        perm_b = all_animals[n_a:]

        # Fixed denominators per group (see rationale above)
        est_a = aggregate_all_and_fit(perm_a, parameter, trajectory_agg,
                                      explicit_denominator=denom_a)
        est_b = aggregate_all_and_fit(perm_b, parameter, trajectory_agg,
                                      explicit_denominator=denom_b)

        if not np.isnan(est_a) and not np.isnan(est_b):
            perm_diff = est_a - est_b
            perm_stats.append(abs(perm_diff))

    if len(perm_stats) > 0:
        return max(perm_stats)
    else:
        return None



def _permutation_worker_baseline(args):
    """
    Worker function for CLASSICAL permutation test on OFF-food baselines (no curve fitting).
    
    Args:
        args: (perm_idx, all_animals_copy, n_a)
    
    Returns:
        null_diff or None if computation failed
    """
    perm_idx, all_animals_copy, n_a = args
    
    # Shuffle animals WITHOUT replacement
    np.random.shuffle(all_animals_copy)
    
    # Split into two groups of original sizes
    perm_a = all_animals_copy[:n_a]
    perm_b = all_animals_copy[n_a:]
    
    # Aggregate baselines (no fitting)
    mean_a = aggregate_baselines_mean(perm_a)
    mean_b = aggregate_baselines_mean(perm_b)
    
    if not np.isnan(mean_a) and not np.isnan(mean_b):
        return mean_a - mean_b
    else:
        return None


def _permutation_worker_baseline_maxt(args):
    """
    Worker function for CLASSICAL Max-T permutation test on OFF-food baselines.
    
    Args:
        args: (perm_idx, comparisons_data)
        comparisons_data: List of (group_a_animals, group_b_animals) tuples
    
    Returns:
        max_stat or None if all computations failed
    """
    perm_idx, comparisons_data = args
    
    perm_stats = []
    
    for group_a_animals, group_b_animals in comparisons_data:
        # Pool and shuffle WITHOUT replacement
        all_animals = group_a_animals + group_b_animals
        np.random.shuffle(all_animals)
        
        n_a = len(group_a_animals)
        perm_a = all_animals[:n_a]
        perm_b = all_animals[n_a:]
        
        # Aggregate baselines
        mean_a = aggregate_baselines_mean(perm_a)
        mean_b = aggregate_baselines_mean(perm_b)
        
        if not np.isnan(mean_a) and not np.isnan(mean_b):
            perm_diff = mean_a - mean_b
            perm_stats.append(abs(perm_diff))
    
    if len(perm_stats) > 0:
        return max(perm_stats)
    else:
        return None

# =============================================================================
# Core Permutation Functions
# =============================================================================

def logistic_curve(t, y_initial, y_final, B, M):
    """
    Standard 4-parameter logistic curve (formerly Richards curve with nu=1).
    Copied from richards_speed_analysis_COMPLETE.py
    """
    A = y_final - y_initial
    with np.errstate(over="ignore", invalid="ignore"):
        denom = 1.0 + np.exp(-B * (t - M))
    denom = np.clip(denom, 1e-10, 1e10)
    return y_initial + A / denom


def fit_single_robust(time_rel: np.ndarray, speed: np.ndarray,
                      loss: str = 'huber') -> dict:
    """
    Fit a single logistic curve using robust loss function.

    Args:
        time_rel: Time values (relative to food encounter)
        speed: Speed values
        loss: Loss function ('least_squares', 'huber', 'soft_l1', 'cauchy')

    Returns:
        Dictionary with fitted parameters and metrics
    """
    from scipy.optimize import least_squares

    # Remove NaN values
    valid = ~(np.isnan(time_rel) | np.isnan(speed))
    t = time_rel[valid]
    y = speed[valid]

    if len(t) < 8:
        return {
            'y_initial': np.nan, 'y_final': np.nan, 'A': np.nan,
            'B': np.nan, 'M': np.nan,
            'converged': False, 'r_squared': np.nan
        }

    # Initial guesses using robust percentiles
    yi = np.nanpercentile(y, 10)
    yf = np.nanpercentile(y, 90)
    p0 = [yi, yf, 0.1, 0.0]

    # Robust bounds
    y_p95 = np.percentile(y, 95)
    y_p05 = np.percentile(y, 5)
    y_range = y_p95 - y_p05

    lower = [max(0, y_p05 - 0.2 * y_range), max(0, y_p05 - 0.2 * y_range),
             1e-3, np.min(t) - 60]
    upper = [y_p95 + 0.2 * y_range, y_p95 + 0.2 * y_range,
             10.0, np.max(t) + 60]

    # Define residual function for robust fitting
    def residuals(params, t, y):
        return logistic_curve(t, *params) - y

    try:
        # Robust fitting with specified loss function
        result = least_squares(
            residuals, p0,
            loss=loss,
            bounds=(lower, upper),
            max_nfev=20000,
            args=(t, y)
        )

        params = result.x
        yhat = logistic_curve(t, *params)

        # Calculate R² (using standard squared residuals for interpretability)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        yi, yf, B, M = params

        return {
            'y_initial': yi,
            'y_final': yf,
            'A': yf - yi,
            'B': B,
            'M': M,
            'converged': result.success,
            'r_squared': r2
        }
    except Exception:
        return {
            'y_initial': np.nan, 'y_final': np.nan, 'A': np.nan,
            'B': np.nan, 'M': np.nan,
            'converged': False, 'r_squared': np.nan
        }


def fit_single(time_rel: np.ndarray, speed: np.ndarray) -> dict:
    """
    Wrapper maintaining backward compatibility.
    Now uses robust Huber loss by default.
    """
    return fit_single_robust(time_rel, speed, loss='huber')


def estimate_parameter_single_bag(
    animal_dataframes: List[pd.DataFrame],
    parameter: str = 'y_initial',
    bag_size: Optional[int] = None,  # If None, uses actual group size (classical bootstrap)
    trajectory_agg: str = 'median',
    off_food_by_animal: Optional[Dict[Tuple[int, int], float]] = None,
    off_food_group_mean: Optional[Dict[Tuple[str, str, str], float]] = None,
    on_food_y_initial_group_mean: Optional[Dict[Tuple[str, str, str], float]] = None
) -> float:
    """
    Estimate a parameter using a pointwise aggregated trajectory (median OR mean) with robust fitting.

    This matches the bootstrap estimator used in the Richards script:
      1) Sample animals with replacement (bagging)
      2) Align to a common time grid
      3) Aggregate pointwise across animals (median or mean)
      4) Robust Richards fit (Huber loss)

    Behavioral context:
      - food_detection_score = y_initial / mean(OFF-food baseline for that GROUP)
        (GROUP mean baseline matches the Richards behavioral-context definition.)

    Args:
        animal_dataframes: List of animal time-series DataFrames. Each must have:
            columns: ['time_rel', 'speed']
            attrs:
              - animal_id: (assay_num, track_num)
              - group_key: (treatment, sex, strain_genotype)
        parameter: Parameter name to extract.
        bag_size: Number of animals sampled with replacement per bag.
                  If None, uses len(animal_dataframes) (classical bootstrap).
        trajectory_agg: 'median' or 'mean' (case-insensitive).
        off_food_by_animal: dict keyed by (assay_num, track_num) -> baseline_speed.
            Only required for validating animal coverage when computing food_detection_score.
        off_food_group_mean: dict keyed by (treatment, sex, strain_genotype) -> mean baseline_speed.
            REQUIRED for food_detection_score.
        on_food_y_initial_group_mean: dict keyed by (treatment, sex, strain_genotype) -> mean y_initial.
            REQUIRED for food_encounter_score. Pre-computed from ON-food bootstrap/individual results.

    Returns:
        Estimated parameter value for this bag, or np.nan on failure.
    """
    try:
        if len(animal_dataframes) == 0:
            return np.nan

        # Classical bootstrap: use actual group size if bag_size not specified
        if bag_size is None:
            bag_size = len(animal_dataframes)

        # Sample animals with replacement
        sampled_animals = random.choices(animal_dataframes, k=bag_size)

        # Build a common time grid from the sampled animals
        all_times = []
        for animal_df in sampled_animals:
            t = animal_df['time_rel'].to_numpy()
            all_times.append(t)

        if len(all_times) == 0:
            return np.nan

        # Use an evenly spaced grid spanning the union of times
        t_min = min(np.nanmin(t) for t in all_times if len(t) > 0)
        t_max = max(np.nanmax(t) for t in all_times if len(t) > 0)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            return np.nan

        # Use 1.0s intervals to match video frame rate (not 0.5s or arbitrary 200 points)
        # This matches the Richards script bootstrap implementation
        total_duration = t_max - t_min
        n_grid = int(total_duration / 1.0) + 1  # 1.0s interval
        time_grid = np.linspace(t_min, t_max, n_grid)

        # Interpolate each animal onto the grid
        speeds_at_each_time = []
        for animal_df in sampled_animals:
            t = animal_df['time_rel'].to_numpy()
            s = animal_df['speed'].to_numpy()
            if len(t) < 2:
                continue
            # Sort by time just in case
            order = np.argsort(t)
            t = t[order]
            s = s[order]

            # Interpolate; outside range becomes NaN
            interp = np.interp(time_grid, t, s, left=np.nan, right=np.nan)
            speeds_at_each_time.append(interp)

        if len(speeds_at_each_time) == 0:
            return np.nan

        speeds_array = np.array(speeds_at_each_time)

        agg = (trajectory_agg or 'median').strip().lower()
        if agg not in ('median', 'mean'):
            agg = 'median'

        if agg == 'mean':
            agg_trajectory = np.nanmean(speeds_array, axis=0)
        else:
            agg_trajectory = np.nanmedian(speeds_array, axis=0)

        # Clean NaNs
        valid_mask = ~np.isnan(agg_trajectory)
        if np.sum(valid_mask) < 8:
            return np.nan

        clean_time = time_grid[valid_mask]
        clean_speed = agg_trajectory[valid_mask]

        # Fit curve to aggregated trajectory using robust loss
        fit_result = fit_single_robust(clean_time, clean_speed, loss='huber')
        if not fit_result.get('converged', False):
            return np.nan

        # Behavioral context parameters
        if parameter == 'food_detection_score':
            if off_food_group_mean is None:
                raise ValueError("OFF-food group mean baselines required for food_detection_score")

            # Determine the group key for this bag (should be homogeneous by construction)
            group_key = None
            for animal_df in sampled_animals:
                gk = animal_df.attrs.get('group_key')
                if gk is None:
                    continue
                if group_key is None:
                    group_key = gk
                elif gk != group_key:
                    # Mixed-group bag indicates a logic error upstream
                    return np.nan

            if group_key is None:
                return np.nan

            # Support OFF-food baselines keyed either by (treatment, sex, genotype)
            # or by (sex, genotype) when treatment is ignored.
            try:
                any_key = next(iter(off_food_group_mean.keys()))
                key_len = len(any_key) if hasattr(any_key, '__len__') else 0
            except StopIteration:
                key_len = 0

            if key_len == 2:
                denom = off_food_group_mean.get((group_key[1], group_key[2]), np.nan)
            else:
                denom = off_food_group_mean.get(group_key, np.nan)
            if not np.isfinite(denom) or denom == 0:
                return np.nan

            # Optional coverage check: ensure sampled animals have OFF-food baselines available
            if off_food_by_animal is not None:
                for animal_df in sampled_animals:
                    animal_id = animal_df.attrs.get('animal_id')
                    if animal_id is None or animal_id not in off_food_by_animal:
                        # Missing baseline for an animal; treat as invalid bag
                        return np.nan

            return fit_result['y_initial'] / denom

        elif parameter == 'food_encounter_score':
            # Match Richards script: y_final / mean(y_initial for GROUP)
            # Uses a fixed group mean pre-computed from ON-food bootstrap/individual results.
            if on_food_y_initial_group_mean is None:
                raise ValueError("ON-food y_initial group means required for food_encounter_score")

            group_key = None
            for animal_df in sampled_animals:
                gk = animal_df.attrs.get('group_key')
                if gk is None:
                    continue
                if group_key is None:
                    group_key = gk
                elif gk != group_key:
                    return np.nan  # Mixed-group bag

            if group_key is None:
                return np.nan

            try:
                any_key = next(iter(on_food_y_initial_group_mean.keys()))
                key_len = len(any_key) if hasattr(any_key, '__len__') else 0
            except StopIteration:
                key_len = 0

            if key_len == 2:
                denom = on_food_y_initial_group_mean.get((group_key[1], group_key[2]), np.nan)
            else:
                denom = on_food_y_initial_group_mean.get(group_key, np.nan)

            if not np.isfinite(denom) or denom == 0:
                return np.nan

            yf = fit_result.get('y_final', np.nan)
            if np.isfinite(yf):
                return yf / denom
            return np.nan

        # Standard parameters
        val = fit_result.get(parameter, np.nan)
        if np.isnan(val):
            return np.nan
        return float(val)

    except Exception:
        return np.nan


def aggregate_all_and_fit(
    animal_dataframes: List[pd.DataFrame],
    parameter: str = 'y_initial',
    trajectory_agg: str = 'median',
    explicit_denominator: Optional[float] = None
) -> float:
    """
    CLASSICAL PERMUTATION: Aggregate ALL animals (no sampling) and fit ONE curve.

    This is for TRUE classical permutation testing:
      1) Use ALL animals in the group (no sampling/replacement)
      2) Align to common time grid
      3) Aggregate pointwise across ALL animals (median or mean)
      4) Fit ONE Richards curve
      5) Extract parameter

    For score parameters (food_detection_score, food_encounter_score), the
    denominator must be supplied via explicit_denominator.  It is pre-computed
    per group before the permutation loop so that every permutation iteration
    uses the same fixed denominator for each group, matching the convention used
    when the observed statistic was calculated.

    Args:
        animal_dataframes: List of ALL animals in this permuted group
        parameter: Parameter to extract
        trajectory_agg: 'median' or 'mean'
        explicit_denominator: Pre-computed denominator for food_detection_score only.
            Pass the group's bootstrap-derived mean off-food baseline speed.
            food_encounter_score uses yf/yi from the same fit (see below).
            Ignored for all other parameters.

    Returns:
        Parameter value or np.nan on failure
    """
    try:
        if len(animal_dataframes) == 0:
            return np.nan

        # Use ALL animals (NO SAMPLING)
        all_animals = animal_dataframes

        # Build common time grid
        all_times = []
        for animal_df in all_animals:
            t = animal_df['time_rel'].to_numpy()
            all_times.append(t)

        if len(all_times) == 0:
            return np.nan

        # Time grid spanning all data
        t_min = min(np.nanmin(t) for t in all_times if len(t) > 0)
        t_max = max(np.nanmax(t) for t in all_times if len(t) > 0)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            return np.nan

        # Use 1.0s intervals matching video frame rate
        total_duration = t_max - t_min
        n_grid = int(total_duration / 1.0) + 1
        time_grid = np.linspace(t_min, t_max, n_grid)

        # Interpolate each animal onto grid
        speeds_at_each_time = []
        for animal_df in all_animals:
            t = animal_df['time_rel'].to_numpy()
            s = animal_df['speed'].to_numpy()
            if len(t) < 2:
                continue
            
            order = np.argsort(t)
            t = t[order]
            s = s[order]

            interp = np.interp(time_grid, t, s, left=np.nan, right=np.nan)
            speeds_at_each_time.append(interp)

        if len(speeds_at_each_time) == 0:
            return np.nan

        speeds_array = np.array(speeds_at_each_time)

        # Aggregate across ALL animals
        agg = (trajectory_agg or 'median').strip().lower()
        if agg not in ('median', 'mean'):
            agg = 'median'

        if agg == 'mean':
            agg_trajectory = np.nanmean(speeds_array, axis=0)
        else:
            agg_trajectory = np.nanmedian(speeds_array, axis=0)

        # Clean NaNs
        valid_mask = ~np.isnan(agg_trajectory)
        if np.sum(valid_mask) < 8:
            return np.nan

        clean_time = time_grid[valid_mask]
        clean_speed = agg_trajectory[valid_mask]

        # Fit ONE curve to aggregated trajectory
        fit_result = fit_single_robust(clean_time, clean_speed, loss='huber')
        if not fit_result.get('converged', False):
            return np.nan

        # food_detection_score: fixed denominator is correct because the off-food
        # baseline is from a SEPARATE dataset not being permuted.
        # The null is correctly centered at zero because off-food data is
        # independent of the on-food shuffle.
        if parameter == 'food_detection_score':
            if explicit_denominator is None or not np.isfinite(explicit_denominator) or explicit_denominator == 0:
                return np.nan
            return fit_result['y_initial'] / explicit_denominator

        # food_encounter_score: use yf/yi FROM THE SAME FIT, not a fixed denominator.
        # The y_initial denominator is itself derived from the on-food data being
        # permuted. Fixing it creates a null NOT centered at zero when groups have
        # different y_initial values — which inflates the null variance and kills power.
        # With yf/yi from the same fit, under H0 both shuffled groups draw yi and yf
        # from the same pool, so the null is correctly centered at zero.
        if parameter == 'food_encounter_score':
            yi = fit_result.get('y_initial', np.nan)
            yf = fit_result.get('y_final', np.nan)
            if np.isfinite(yi) and np.isfinite(yf) and yi != 0:
                return yf / yi
            return np.nan

        # Standard parameters (no denominator needed)
        val = fit_result.get(parameter, np.nan)
        if np.isnan(val):
            return np.nan
        return float(val)

    except Exception:
        return np.nan


def aggregate_baselines_mean(animal_dataframes: List[pd.DataFrame]) -> float:
    """
    Aggregate baseline_speed values from OFF-food animals (no curve fitting).
    
    Used for permutation tests of OFF-food baselines directly.
    
    Args:
        animal_dataframes: List of animals, each with 'baseline_speed' column
    
    Returns:
        Mean baseline speed or np.nan on failure
    """
    try:
        if len(animal_dataframes) == 0:
            return np.nan
        
        baselines = []
        for animal_df in animal_dataframes:
            if 'baseline_speed' in animal_df.columns:
                val = animal_df['baseline_speed'].iloc[0]
                if np.isfinite(val):
                    baselines.append(float(val))
        
        if len(baselines) == 0:
            return np.nan
        
        return float(np.mean(baselines))
    
    except Exception:
        return np.nan


# =============================================================================
# OFF-Food Baseline Permutation Functions
# =============================================================================

def standard_permutation_test_baseline(
    group_a_animals: List[pd.DataFrame],
    group_b_animals: List[pd.DataFrame],
    observed_diff: float,
    n_permutations: int = 10000,
    n_cores: Optional[int] = None,
    progress_callback=None
) -> PermutationResult:
    """
    CLASSICAL permutation test for OFF-food baselines (no curve fitting).
    
    Each permutation:
      1. Shuffles animals WITHOUT replacement
      2. Splits into groups of original sizes
      3. Calculates mean baseline_speed per group
      4. Computes difference
    
    No curve fitting - just direct mean of baseline speeds.
    """
    all_animals = group_a_animals + group_b_animals
    n_a = len(group_a_animals)

    if n_cores is None:
        max_cores = cpu_count()
        n_cores = max(1, int(max_cores * 0.75))

    if progress_callback:
        progress_callback(0, n_permutations)

    tasks = []
    for perm_idx in range(n_permutations):
        all_animals_copy = all_animals.copy()
        tasks.append((perm_idx, all_animals_copy, n_a))

    null_diffs = []
    with Pool(n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(_permutation_worker_baseline, tasks, chunksize=10)):
            if result is not None:
                null_diffs.append(result)

            if progress_callback and (i % 100 == 0 or i == n_permutations - 1):
                progress_callback(i + 1, n_permutations)

    null_diffs = np.array(null_diffs)
    # +1 correction (Phipson & Smyth): prevents p=0.0 by treating the observed
    # data as one possible permutation. Minimum p = 1/(n_permutations+1).
    p_value = (np.sum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (len(null_diffs) + 1)

    return PermutationResult(
        comparison_name="Baseline permutation",
        observed_diff=observed_diff,
        p_value=p_value,
        null_distribution=null_diffs,
        method='baseline_standard',
        n_permutations=n_permutations
    )


def maxt_permutation_test_baseline(
    comparisons: List[Comparison],
    animals_by_group: Dict[str, Dict],
    n_permutations: int = 10000,
    n_cores: Optional[int] = None,
    progress_callback=None
) -> List[PermutationResult]:
    """
    CLASSICAL Max-T permutation test for OFF-food baselines with FWER control.
    
    Each permutation:
      1. For EACH comparison: shuffles animals WITHOUT replacement
      2. Calculates mean baseline_speed per group
      3. Computes differences
      4. Takes maximum absolute difference
    
    No curve fitting - just direct mean of baseline speeds.
    """
    if n_cores is None:
        max_cores = cpu_count()
        n_cores = max(1, int(max_cores * 0.75))

    comparisons_data = []
    observed_diffs = []

    for comp in comparisons:
        group_a_animals = animals_by_group[comp.group_a]['animals']
        group_b_animals = animals_by_group[comp.group_b]['animals']
        comparisons_data.append((group_a_animals, group_b_animals))
        observed_diffs.append(comp.observed_diff)

    observed_diffs = np.array(observed_diffs)

    if progress_callback:
        progress_callback(0, n_permutations)

    tasks = [(i, comparisons_data) for i in range(n_permutations)]

    max_null_stats = []
    with Pool(n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(_permutation_worker_baseline_maxt, tasks, chunksize=10)):
            if result is not None:
                max_null_stats.append(result)

            if progress_callback and (i % 100 == 0 or i == n_permutations - 1):
                progress_callback(i + 1, n_permutations)

    max_null_stats = np.array(max_null_stats)

    results = []
    for i, comp in enumerate(comparisons):
        # +1 correction (Phipson & Smyth): prevents p=0.0
        p_value = (np.sum(max_null_stats >= abs(observed_diffs[i])) + 1) / (len(max_null_stats) + 1)
        results.append(PermutationResult(
            comparison_name=comp.name,
            observed_diff=comp.observed_diff,
            p_value=p_value,
            null_distribution=max_null_stats,
            method='baseline_maxt',
            n_permutations=n_permutations
        ))

    return results


def _resolve_denominator(group_animals: List[pd.DataFrame],
                         parameter: str,
                         off_food_group_mean: Optional[Dict] = None,
                         on_food_y_initial_group_mean: Optional[Dict] = None) -> float:
    """
    Resolve the fixed denominator for a score parameter given a group's animals.

    The group_key is taken from the first animal's attrs (all animals in a proper
    group share the same key).  Returns np.nan for non-score parameters (caller
    should pass explicit_denominator=np.nan, which aggregate_all_and_fit ignores).

    Args:
        group_animals: List of animal DataFrames (each has attrs['group_key'])
        parameter: Parameter name being tested
        off_food_group_mean: Dict keyed by group -> mean off-food baseline speed.
            Used for food_detection_score. Should be derived from bootstrap
            replicates (off_food_bootstrap_baselines.csv) for consistency with
            the observed statistic.
        on_food_y_initial_group_mean: Dict keyed by group -> mean y_initial.
            Used for food_encounter_score.

    Returns:
        Scalar denominator for food_detection_score, or np.nan for all other
        parameters (including food_encounter_score, which uses yf/yi from the
        same fit rather than a pre-computed fixed denominator).
    """
    if parameter != 'food_detection_score':
        # food_encounter_score uses yf/yi from the same fit (not a fixed denominator)
        # All other parameters need no denominator at all
        return np.nan

    if len(group_animals) == 0:
        return np.nan

    group_key = group_animals[0].attrs.get('group_key')
    if group_key is None:
        return np.nan

    # Only food_detection_score reaches here
    if off_food_group_mean is None:
        return np.nan
    lookup = off_food_group_mean

    # Support dicts keyed by (treatment, sex, genotype) or (sex, genotype)
    try:
        any_key = next(iter(lookup.keys()))
        key_len = len(any_key) if hasattr(any_key, '__len__') else 0
    except StopIteration:
        return np.nan

    if key_len == 2:
        denom = lookup.get((group_key[1], group_key[2]), np.nan)
    else:
        denom = lookup.get(group_key, np.nan)

    return float(denom) if np.isfinite(denom) and denom != 0 else np.nan


def standard_permutation_test(group_a_animals: List[pd.DataFrame],
                              group_b_animals: List[pd.DataFrame],
                              observed_diff: float,
                              parameter: str = 'y_initial',
                              n_permutations: int = 10000,
                              n_cores: Optional[int] = None,
                              trajectory_agg: str = 'median',
                              off_food_by_animal: Optional[Dict[Tuple[int, int], float]] = None,
                              off_food_group_mean: Optional[Dict[Tuple[str, str, str], float]] = None,
                              on_food_y_initial_group_mean: Optional[Dict[Tuple[str, str, str], float]] = None,
                              progress_callback=None) -> PermutationResult:
    """
    CLASSICAL permutation test for one pairwise comparison.

    Each permutation:
      1. Shuffles animals WITHOUT replacement
      2. Splits into groups of original sizes
      3. Aggregates ALL animals per group (no sampling)
      4. Fits ONE Richards curve per group
      5. Divides by the FIXED group denominator (pre-computed once, not re-derived
         from whichever animals happen to land in each shuffled sub-group)
      6. Computes difference

    The fixed-denominator approach ensures the null distribution is generated
    with the same scaling convention as the observed statistic, giving a tighter
    (less inflated) null and therefore greater power for score parameters.

    Args:
        n_cores: Number of CPU cores to use (None = use 75% of available)
        off_food_group_mean: Bootstrap-derived off-food baseline means per group.
            Used as denominator for food_detection_score.
        on_food_y_initial_group_mean: Bootstrap-derived on-food y_initial means per group.
            Used as denominator for food_encounter_score.
    """
    all_animals = group_a_animals + group_b_animals
    n_a = len(group_a_animals)

    # Pre-compute fixed denominators for each group ONCE before the loop.
    # This is the key correctness fix: every permutation iteration uses the
    # same denom_a for the first-slot group and denom_b for the second-slot
    # group, regardless of which animals were shuffled into each slot.
    denom_a = _resolve_denominator(group_a_animals, parameter,
                                   off_food_group_mean, on_food_y_initial_group_mean)
    denom_b = _resolve_denominator(group_b_animals, parameter,
                                   off_food_group_mean, on_food_y_initial_group_mean)

    # Determine number of cores
    if n_cores is None:
        max_cores = cpu_count()
        n_cores = max(1, int(max_cores * 0.75))

    if progress_callback:
        progress_callback(0, n_permutations)

    # Build task list — each worker gets its own copy of the pooled animal list
    # plus the two fixed denominators.
    tasks = []
    for perm_idx in range(n_permutations):
        all_animals_copy = all_animals.copy()
        tasks.append((perm_idx, all_animals_copy, n_a, parameter, trajectory_agg,
                      denom_a, denom_b))

    # Run parallel permutations
    null_diffs = []
    with Pool(n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(_permutation_worker_standard, tasks, chunksize=10)):
            if result is not None:
                null_diffs.append(result)

            if progress_callback and (i % 100 == 0 or i == n_permutations - 1):
                progress_callback(i + 1, n_permutations)

    null_diffs = np.array(null_diffs)
    # +1 correction (Phipson & Smyth): prevents p=0.0 by treating the observed
    # data as one possible permutation. Minimum p = 1/(n_permutations+1).
    p_value = (np.sum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (len(null_diffs) + 1)

    return PermutationResult(
        comparison_name="Standard permutation",
        observed_diff=observed_diff,
        p_value=p_value,
        null_distribution=null_diffs,
        method='standard',
        n_permutations=n_permutations
    )


def maxt_permutation_test(comparisons: List[Comparison],
                          animals_by_group: Dict[str, List[pd.DataFrame]],
                          parameter: str = 'y_initial',
                          n_permutations: int = 10000,
                          n_cores: Optional[int] = None,
                          trajectory_agg: str = 'median',
                          off_food_by_animal: Optional[Dict[Tuple[int, int], float]] = None,
                          off_food_group_mean: Optional[Dict[Tuple[str, str, str], float]] = None,
                          on_food_y_initial_group_mean: Optional[Dict[Tuple[str, str, str], float]] = None,
                          progress_callback=None) -> List[PermutationResult]:
    """
    CLASSICAL Max-T permutation test for multiple comparisons with FWER control.

    Each permutation:
      1. For EACH comparison: shuffles animals WITHOUT replacement
      2. Splits into groups of original sizes
      3. Aggregates ALL animals per group (no sampling)
      4. Fits ONE Richards curve per group
      5. Divides by the FIXED group denominator (pre-computed once per comparison)
      6. Computes difference
      7. Tracks MAXIMUM absolute difference across all comparisons

    Fixed denominators per group (see standard_permutation_test for rationale).
    Provides strong family-wise error rate (FWER) control.

    Args:
        n_cores: Number of CPU cores to use (None = use 75% of available)
        off_food_group_mean: Bootstrap-derived off-food baseline means.
            Denominator for food_detection_score.
        on_food_y_initial_group_mean: Bootstrap-derived on-food y_initial means.
            Denominator for food_encounter_score.
    """
    observed_diffs = [comp.observed_diff for comp in comparisons]

    # Determine number of cores
    if n_cores is None:
        max_cores = cpu_count()
        n_cores = max(1, int(max_cores * 0.75))

    if progress_callback:
        progress_callback(0, n_permutations)

    # Pre-compute fixed denominators for each group in each comparison ONCE.
    # comparisons_data entries are (group_a_animals, group_b_animals, denom_a, denom_b)
    comparisons_data = []
    for comp in comparisons:
        group_a_animals = animals_by_group[comp.group_a]
        group_b_animals = animals_by_group[comp.group_b]
        denom_a = _resolve_denominator(group_a_animals, parameter,
                                       off_food_group_mean, on_food_y_initial_group_mean)
        denom_b = _resolve_denominator(group_b_animals, parameter,
                                       off_food_group_mean, on_food_y_initial_group_mean)
        comparisons_data.append((group_a_animals, group_b_animals, denom_a, denom_b))

    # Build task list — comparisons_data is read-only so safe to share across tasks
    tasks = []
    for perm_idx in range(n_permutations):
        tasks.append((perm_idx, comparisons_data, parameter, trajectory_agg))

    # Run parallel permutations
    max_null_stats = []
    with Pool(n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(_permutation_worker_maxt, tasks, chunksize=10)):
            if result is not None:
                max_null_stats.append(result)

            if progress_callback and (i % 100 == 0 or i == n_permutations - 1):
                progress_callback(i + 1, n_permutations)

    max_null_stats = np.array(max_null_stats)

    results = []
    for i, comp in enumerate(comparisons):
        # +1 correction (Phipson & Smyth): prevents p=0.0
        p_value = (np.sum(max_null_stats >= abs(observed_diffs[i])) + 1) / (len(max_null_stats) + 1)
        results.append(PermutationResult(
            comparison_name=comp.name,
            observed_diff=comp.observed_diff,
            p_value=p_value,
            null_distribution=max_null_stats,
            method='maxt',
            n_permutations=n_permutations
        ))

    return results


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_bootstrap_results(bootstrap_csv_path: Path, parameter: str = 'y_initial') -> pd.DataFrame:
    """Load bootstrap results from richards curve fitting."""
    df = pd.read_csv(bootstrap_csv_path)

    if 'converged' in df.columns:
        df = df[df['converged'] == True].copy()

    return df


def organize_animals_by_group(windowed_csv_path: Path,
                              group_columns: List[str] = ['treatment', 'sex', 'strain_genotype']) -> Dict[str, Dict]:
    """
    Organize animals into groups with both raw data and metadata.
    Each animal DataFrame gets an 'animal_id' attribute storing (assay_num, track_num).
    """
    df = pd.read_csv(windowed_csv_path)

    groups = {}

    for group_vals, group_df in df.groupby(group_columns):
        group_label = '|'.join([str(v) for v in group_vals])

        animals = []
        for (assay, track), animal_df in group_df.groupby(['assay_num', 'track_num']):
            animal_data = animal_df[['time_rel', 'speed']].copy()
            animal_data.attrs['animal_id'] = (int(assay), int(track))
            # Store the group key tuple so bag estimators can access GROUP-level denominators
            animal_data.attrs['group_key'] = tuple(str(v) for v in group_vals)
            animals.append(animal_data)

        groups[group_label] = {
            'animals': animals,
            'metadata': dict(zip(group_columns, group_vals)),
            'n_animals': len(animals)
        }

    return groups


def organize_off_food_baselines_by_group(off_food_csv_path: Path,
                                         group_columns: List[str] = ['treatment', 'sex', 'strain_genotype']) -> Dict[str, Dict]:
    """
    Organize OFF-food baseline animals into groups for permutation testing.
    Each animal is represented as a simple dict with baseline_speed.
    """
    df = pd.read_csv(off_food_csv_path)
    
    # Validate required columns
    required = ['assay_num', 'track_num', 'baseline_speed'] + group_columns
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"OFF-food CSV missing columns: {missing}")
    
    groups = {}
    
    for group_vals, group_df in df.groupby(group_columns):
        group_label = '|'.join([str(v) for v in group_vals])
        
        # Each animal is a simple container with baseline_speed and ID
        animals = []
        for _, row in group_df.iterrows():
            animal_data = pd.DataFrame({'baseline_speed': [row['baseline_speed']]})
            animal_data.attrs['animal_id'] = (int(row['assay_num']), int(row['track_num']))
            animal_data.attrs['group_key'] = tuple(str(v) for v in group_vals)
            animals.append(animal_data)
        
        groups[group_label] = {
            'animals': animals,
            'metadata': dict(zip(group_columns, group_vals)),
            'n_animals': len(animals)
        }
    
    return groups


def get_observed_differences_from_bootstrap(bootstrap_df: pd.DataFrame,
                                           comparisons: List[Tuple[str, str, str]],
                                           parameter: str = 'y_initial',
                                           group_columns: List[str] = ['treatment', 'sex', 'strain_genotype']) -> List[Comparison]:
    """
    Calculate observed differences from bootstrap results using MEDIAN (robust).

    Changed from mean to median for robustness to outliers in bootstrap distribution.
    """
    comparison_objects = []

    for name, group_a_label, group_b_label in comparisons:
        group_a_vals = group_a_label.split('|')
        group_b_vals = group_b_label.split('|')

        mask_a = np.ones(len(bootstrap_df), dtype=bool)
        mask_b = np.ones(len(bootstrap_df), dtype=bool)

        for col, val_a, val_b in zip(group_columns, group_a_vals, group_b_vals):
            mask_a &= (bootstrap_df[col] == val_a)
            mask_b &= (bootstrap_df[col] == val_b)

        group_a_data = bootstrap_df[mask_a][parameter].values
        group_b_data = bootstrap_df[mask_b][parameter].values

        if len(group_a_data) == 0 or len(group_b_data) == 0:
            continue

        median_a = np.median(group_a_data)
        median_b = np.median(group_b_data)
        obs_diff = median_a - median_b

        comparison_objects.append(Comparison(
            name=name,
            group_a=group_a_label,
            group_b=group_b_label,
            observed_diff=obs_diff,
            group_a_mean=median_a,  # attribute name kept for compatibility
            group_b_mean=median_b
        ))

    return comparison_objects


# =============================================================================
# Post-hoc Correction Methods
# =============================================================================

def bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni correction for multiple comparisons."""
    return np.minimum(p_values * len(p_values), 1.0)


def holm_bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_idx[i]] = min(p * (n - i), 1.0)

    for i in range(1, n):
        adjusted[sorted_idx[i]] = max(adjusted[sorted_idx[i]], adjusted[sorted_idx[i - 1]])

    return adjusted


def fdr_benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[sorted_idx[i]] = min(sorted_p[i] * n / (i + 1), 1.0)

    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])

    return adjusted


def bootstrap_one_sample_test(
    bootstrap_scores: np.ndarray,
    group_label: str,
    parameter: str,
    mu0: float = 1.0,
) -> OneSampleResult:
    """
    One-sample test using the bootstrap distribution directly as the sampling
    distribution of the estimator.

    Each bootstrap replicate already resampled animals with replacement, so the
    spread of the distribution reflects animal-level variability — not replicate
    count.  No SEM is computed; the CI is purely the 2.5/97.5 percentiles of the
    bootstrap distribution.

    Primary readout: does mu0 fall outside the 95% CI?
      - ci_lo > mu0  →  group is significantly ABOVE mu0
      - ci_hi < mu0  →  group is significantly BELOW mu0
      - ci_lo <= mu0 <= ci_hi  →  not distinguishable from mu0

    The p_value is a continuous version of the same question: how far into the
    tail does mu0 sit, doubled for two-tailed, with the +1/(n+1) floor so it
    can never be exactly 0.

    Args:
        bootstrap_scores: Array of per-replicate score values.
        group_label: Human-readable label for logging.
        parameter: Parameter name (e.g. 'food_detection_score').
        mu0: Reference value to test against (default 1.0).

    Returns:
        OneSampleResult dataclass.
    """
    scores = bootstrap_scores[np.isfinite(bootstrap_scores)]
    n = len(scores)

    mean_score = float(np.mean(scores))
    ci_lo = float(np.percentile(scores, 2.5))
    ci_hi = float(np.percentile(scores, 97.5))

    # Two-tailed empirical p: fraction of bootstrap dist on the mu0 side,
    # doubled, with +1/(n+1) floor so p can never be exactly 0.
    if mean_score >= mu0:
        p = 2.0 * (np.sum(scores <= mu0) + 1) / (n + 1)
    else:
        p = 2.0 * (np.sum(scores >= mu0) + 1) / (n + 1)
    p = float(min(p, 1.0))

    return OneSampleResult(
        group=group_label,
        parameter=parameter,
        mu0=mu0,
        mean=mean_score,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        mean_minus_mu0=mean_score - mu0,
        p_value=p,
        significant=not (ci_lo <= mu0 <= ci_hi),
        n_bootstrap=n,
    )




# =============================================================================
# Unified Package Loading Functions
# =============================================================================

def organize_off_food_baselines_by_group_from_df(
    off_food_df: pd.DataFrame,
    group_columns: List[str] = ['treatment', 'sex', 'strain_genotype']
) -> Dict[str, Dict]:
    """
    Organize OFF-food baseline animals into groups from DataFrame.
    
    Args:
        off_food_df: DataFrame with columns [assay_num, track_num, baseline_speed, 
                     treatment, sex, strain_genotype]
    
    Returns:
        Dictionary: {group_label: {'animals': [...], 'metadata': {...}, 'n_animals': int}}
    """
    groups = {}
    
    for group_vals, group_df in off_food_df.groupby(group_columns):
        group_label = '|'.join([str(v) for v in group_vals])
        
        animals = []
        for _, row in group_df.iterrows():
            animal_data = pd.DataFrame({'baseline_speed': [row['baseline_speed']]})
            animal_data.attrs['animal_id'] = (int(row['assay_num']), int(row['track_num']))
            animal_data.attrs['group_key'] = tuple(str(v) for v in group_vals)
            animals.append(animal_data)
        
        groups[group_label] = {
            'animals': animals,
            'metadata': dict(zip(group_columns, group_vals)),
            'n_animals': len(animals)
        }
    
    return groups


def get_observed_differences_unified(
    bootstrap_df: pd.DataFrame,
    comparisons: List[Tuple[str, str, str]],
    parameter: str,
    data_type: str = 'on_food',
    group_columns: List[str] = ['treatment', 'sex', 'strain_genotype']
) -> List[Comparison]:
    """
    Calculate observed differences from combined bootstrap results.
    Works for both ON-food and OFF-food parameters.
    
    Args:
        bootstrap_df: Combined bootstrap results with columns:
                     [data_type, bootstrap_replicate, treatment, sex, 
                      strain_genotype, parameter, value]
        comparisons: List of (name, group_a_label, group_b_label) tuples
        parameter: Parameter to test (e.g., 'y_initial', 'baseline_speed')
        data_type: 'on_food' or 'off_food'
    
    Returns:
        List of Comparison objects with observed differences
    """
    df_filtered = bootstrap_df[
        (bootstrap_df['data_type'] == data_type) &
        (bootstrap_df['parameter'] == parameter)
    ].copy()
    
    if len(df_filtered) == 0:
        raise ValueError(f"No bootstrap data found for {data_type} / {parameter}")
    
    comparison_objects = []
    
    for name, group_a_label, group_b_label in comparisons:
        group_a_vals = group_a_label.split('|')
        group_b_vals = group_b_label.split('|')
        
        mask_a = np.ones(len(df_filtered), dtype=bool)
        mask_b = np.ones(len(df_filtered), dtype=bool)
        
        for i, col in enumerate(group_columns):
            mask_a &= (df_filtered[col] == group_a_vals[i])
            mask_b &= (df_filtered[col] == group_b_vals[i])
        
        values_a = df_filtered[mask_a]['value'].values
        values_b = df_filtered[mask_b]['value'].values
        
        if len(values_a) == 0 or len(values_b) == 0:
            continue
        
        median_a = np.median(values_a)
        median_b = np.median(values_b)
        obs_diff = median_a - median_b
        
        comparison_objects.append(Comparison(
            name=name,
            group_a=group_a_label,
            group_b=group_b_label,
            observed_diff=obs_diff,
            group_a_mean=median_a,
            group_b_mean=median_b
        ))
    
    return comparison_objects


# =============================================================================
# GUI Application
# =============================================================================

class PermutationTestingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Permutation Testing Application")

        # Window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.8))
        window_height = min(900, int(screen_height * 0.85))

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 700)

        # Data storage
        self.bootstrap_df = None
        self.groups = None
        self.animals_by_group = None
        self.off_food_df = None  # OFF-food baselines DataFrame
        self.off_food_by_animal = None  # Dict: (assay, track) -> baseline_speed
        self.off_food_group_mean = None  # Dict: (treatment, sex, strain_genotype) -> mean baseline_speed
        self.on_food_y_initial_group_mean = None  # Dict: (treatment, sex, strain_genotype) -> mean y_initial
        self.selected_comparisons = []
        self.results = []
        self.comparisons = []
        self.one_sample_results = []   # results from one-sample CI tests
        self.output_dir = None

        # Unified package loading support
        self.permutation_package_dir = None
        self.bootstrap_combined_df = None  # Combined ON+OFF food bootstrap results
        self.off_food_individual_df = None  # OFF-food individual baselines
        self.off_food_groups = None  # OFF-food animals organized by group

        # Parameters
        self.parameter = tk.StringVar(value='y_initial')
        self.test_method = tk.StringVar(value='maxt')
        self.correction_method = tk.StringVar(value='none')
        self.n_permutations = tk.IntVar(value=10000)
        self.bag_size = tk.IntVar(value=10)
        self.trajectory_agg = tk.StringVar(value='median')  # pointwise aggregation for bag trajectory
        self.alpha = tk.DoubleVar(value=0.05)

        # CPU cores (default to 75% of available)
        max_cores = cpu_count()
        default_cores = max(1, int(max_cores * 0.75))
        self.n_cores = tk.IntVar(value=default_cores)
        self.max_cores = max_cores

        self.setup_gui()

    def setup_gui(self):
        # Main container
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        # Configure rows: main content gets 2x weight of results (2:1 ratio)
        main.rowconfigure(1, weight=2)  # Main content area
        main.rowconfigure(3, weight=1)  # Results section

        # Title
        ttk.Label(main, text="Permutation Testing for Bootstrap Bagging Analysis",
                  font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

        # LEFT COLUMN — scrollable so all sections are reachable on small screens
        left_outer = ttk.Frame(main)
        left_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        left_canvas.grid(row=0, column=0, sticky="nsew")

        left_scroll = ttk.Scrollbar(left_outer, orient="vertical",
                                    command=left_canvas.yview)
        left_scroll.grid(row=0, column=1, sticky="ns")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_frame = ttk.Frame(left_canvas)
        left_frame.columnconfigure(0, weight=1)

        left_canvas_window = left_canvas.create_window(
            (0, 0), window=left_frame, anchor="nw")

        def _on_left_frame_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_left_canvas_configure(event):
            left_canvas.itemconfig(left_canvas_window, width=event.width)

        left_frame.bind("<Configure>", _on_left_frame_configure)
        left_canvas.bind("<Configure>", _on_left_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)           # Windows/macOS
            widget.bind("<Button-4>",
                        lambda e: left_canvas.yview_scroll(-1, "units"))  # Linux scroll up
            widget.bind("<Button-5>",
                        lambda e: left_canvas.yview_scroll(1, "units"))   # Linux scroll down
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        left_frame.bind("<Map>", lambda e: _bind_mousewheel(left_frame))

        self.create_file_section(left_frame)
        self.create_parameters_section(left_frame)
        self.create_method_section(left_frame)

        # RIGHT COLUMN
        right_frame = ttk.Frame(main)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.create_comparison_section(right_frame)

        # Action buttons
        self.create_action_section(main)

        # Log/Results section
        self.create_results_section(main)

    def create_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Data Files", padding="10")
        frame.pack(fill="x", pady=(0, 5))
        frame.columnconfigure(1, weight=1)

        # PACKAGE LOADING (RECOMMENDED)
        ttk.Label(frame, text="Permutation Package:", font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=3)
        pkg_label = ttk.Label(frame, text="(Recommended - load one directory)", foreground="blue", font=("", 8))
        pkg_label.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load Package", command=self.load_permutation_package, width=15).grid(
            row=0, column=2, padx=3)
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)
        
        # INDIVIDUAL FILE LOADING (LEGACY)
        ttk.Label(frame, text="Or load individual files:", font=("Helvetica", 8)).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(0,5))

        ttk.Label(frame, text="Bootstrap CSV:").grid(row=3, column=0, sticky="w", pady=3)
        self.bootstrap_label = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.bootstrap_label.grid(row=3, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load", command=self.load_bootstrap, width=10).grid(row=3, column=2, padx=3)

        ttk.Label(frame, text="Animal CSV:").grid(row=4, column=0, sticky="w", pady=3)
        self.animal_label = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.animal_label.grid(row=4, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load", command=self.load_animals, width=10).grid(row=4, column=2, padx=3)

        ttk.Label(frame, text="OFF-food CSV:").grid(row=5, column=0, sticky="w", pady=3)
        self.off_food_label = ttk.Label(frame, text="Optional (for behavioral context)", foreground="gray")
        self.off_food_label.grid(row=5, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load", command=self.load_off_food, width=10).grid(row=5, column=2, padx=3)

        ttk.Label(frame, text="Output dir:").grid(row=6, column=0, sticky="w", pady=3)
        self.output_label = ttk.Label(frame, text="Not selected", foreground="gray")
        self.output_label.grid(row=6, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Select", command=self.select_output_dir, width=10).grid(row=6, column=2, padx=3)

    def create_parameters_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Test Parameters", padding="10")
        frame.pack(fill="x", pady=(0, 5))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Parameter:").grid(row=0, column=0, sticky="w", pady=3)
        param_combo = ttk.Combobox(frame, textvariable=self.parameter, width=20,
                                   values=['y_initial', 'y_final', 'A', 'B', 'M',
                                           'food_detection_score', 'food_encounter_score',
                                           'baseline_speed'])
        param_combo.grid(row=0, column=1, sticky="w", padx=3)

        ttk.Label(frame, text="Permutations:").grid(row=1, column=0, sticky="w", pady=3)
        perm_frame = ttk.Frame(frame)
        perm_frame.grid(row=1, column=1, sticky="w", padx=3)
        ttk.Entry(perm_frame, textvariable=self.n_permutations, width=10).pack(side="left")
        ttk.Label(perm_frame, text="(10000 recommended, 5000 for quick test)",
                  foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="Method:").grid(row=2, column=0, sticky="w", pady=3)
        method_frame = ttk.Frame(frame)
        method_frame.grid(row=2, column=1, sticky="w", padx=3)
        ttk.Label(method_frame, text="Classical permutation test", 
                  foreground="black", font=("", 9)).pack(side="left")
        ttk.Label(method_frame, text="(aggregates ALL animals per group, no sampling)",
                  foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="Trajectory aggregation:").grid(row=3, column=0, sticky="w", pady=3)
        agg_frame = ttk.Frame(frame)
        agg_frame.grid(row=3, column=1, sticky="w", padx=3)
        agg_combo = ttk.Combobox(agg_frame, textvariable=self.trajectory_agg, values=["median", "mean"], width=12, state="readonly")
        agg_combo.pack(side="left")
        ttk.Label(agg_frame, text="(pointwise across animals in each bag)", foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="CPU cores:").grid(row=4, column=0, sticky="w", pady=3)
        cores_frame = ttk.Frame(frame)
        cores_frame.grid(row=4, column=1, sticky="w", padx=3)
        ttk.Entry(cores_frame, textvariable=self.n_cores, width=10).pack(side="left")
        ttk.Label(cores_frame, text=f"(max {self.max_cores}, using {self.n_cores.get()} = 75%)",
                  foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="Alpha:").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.alpha, width=10).grid(row=4, column=1, sticky="w", padx=3)

    def create_method_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Test Method", padding="10")
        frame.pack(fill="x", pady=(0, 5))

        ttk.Label(frame, text="Method:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=3)
        ttk.Radiobutton(frame, text="Max-T (Built-in FWER control - RECOMMENDED)",
                        variable=self.test_method, value='maxt').pack(anchor="w", padx=20)
        ttk.Radiobutton(frame, text="Standard (Requires post-hoc correction)",
                        variable=self.test_method, value='standard').pack(anchor="w", padx=20)

        ttk.Label(frame, text="Post-hoc correction:",
                  font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 3))
        ttk.Label(frame,
                  text="Note: BH FDR applied to Max-T p-values is more conservative\n"
                       "than Standard + FDR (double correction). Use Standard + FDR\n"
                       "for maximum power.",
                  foreground="gray", font=("", 8), justify="left").pack(anchor="w", padx=20)
        ttk.Radiobutton(frame, text="None",
                        variable=self.correction_method, value='none').pack(anchor="w", padx=20)
        ttk.Radiobutton(frame, text="Bonferroni",
                        variable=self.correction_method, value='bonferroni').pack(anchor="w", padx=20)
        ttk.Radiobutton(frame, text="Holm-Bonferroni",
                        variable=self.correction_method, value='holm').pack(anchor="w", padx=20)
        ttk.Radiobutton(frame, text="FDR (Benjamini-Hochberg)  \u2190 more power",
                        variable=self.correction_method, value='fdr').pack(anchor="w", padx=20)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(10, 5))

        ttk.Label(frame, text="One-Sample Test \u03bc\u2080 (reference value):",
                  font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 3))
        ttk.Label(frame,
                  text="Used by 'Run One-Sample Tests' to test whether detection/encounter\n"
                       "score distributions exclude \u03bc\u2080 (default 1.0 = no change).\n"
                       "Uses bootstrap 95% CI directly \u2014 no SEM, no t-test.",
                  foreground="gray", font=("", 8), justify="left").pack(anchor="w", padx=20)
        mu0_frame = ttk.Frame(frame)
        mu0_frame.pack(anchor="w", padx=20, pady=(3, 0))
        ttk.Label(mu0_frame, text="\u03bc\u2080:").pack(side="left")
        self.mu0_var = tk.DoubleVar(value=1.0)
        ttk.Entry(mu0_frame, textvariable=self.mu0_var, width=8).pack(side="left", padx=5)

    def create_comparison_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Comparisons", padding="10")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        groups_frame = ttk.Frame(frame)
        groups_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        groups_frame.columnconfigure(1, weight=1)

        ttk.Label(groups_frame, text="Available groups:",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.groups_display = ttk.Label(groups_frame, text="Load data first", foreground="gray")
        self.groups_display.grid(row=0, column=1, sticky="w", padx=5)

        builder_frame = ttk.Frame(frame)
        builder_frame.grid(row=1, column=0, sticky="nsew")
        builder_frame.columnconfigure(0, weight=1)
        builder_frame.columnconfigure(2, weight=1)
        # Row 1 (dropdowns) should not expand - remove weight
        # builder_frame.rowconfigure(1, weight=1)  # REMOVED - was causing dropdowns to be invisible

        ttk.Label(builder_frame, text="Group A:").grid(row=0, column=0, sticky="w", pady=3)
        self.group_a_var = tk.StringVar()
        self.group_a_combo = ttk.Combobox(builder_frame, textvariable=self.group_a_var, width=30)
        self.group_a_combo.grid(row=1, column=0, sticky="ew", padx=3)

        ttk.Label(builder_frame, text="VS", font=("Helvetica", 12, "bold")).grid(row=1, column=1, padx=10)

        ttk.Label(builder_frame, text="Group B:").grid(row=0, column=2, sticky="w", pady=3)
        self.group_b_var = tk.StringVar()
        self.group_b_combo = ttk.Combobox(builder_frame, textvariable=self.group_b_var, width=30)
        self.group_b_combo.grid(row=1, column=2, sticky="ew", padx=3)

        ttk.Label(builder_frame, text="Comparison name (optional):").grid(row=2, column=0, sticky="w", pady=(10, 3))
        self.comp_name_var = tk.StringVar()
        ttk.Entry(builder_frame, textvariable=self.comp_name_var).grid(row=3, column=0, columnspan=3, sticky="ew", padx=3)

        ttk.Button(builder_frame, text="Add Comparison",
                   command=self.add_comparison).grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(builder_frame, text="Browse All…",
                   command=self.open_comparison_browser).grid(row=4, column=2, pady=10, sticky="ew", padx=3)

        ttk.Label(builder_frame, text="Selected Comparisons:",
                  font=("Helvetica", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(10, 3))

        list_frame = ttk.Frame(builder_frame)
        list_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=3)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        builder_frame.rowconfigure(6, weight=1)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.comp_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=10)
        self.comp_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.comp_listbox.yview)

        ttk.Button(builder_frame, text="Remove Selected",
                   command=self.remove_comparison).grid(row=7, column=0, columnspan=3, pady=5)

        # Save / Load comparison lists
        btn_frame = ttk.Frame(builder_frame)
        btn_frame.grid(row=8, column=0, columnspan=3, pady=(0, 5))
        ttk.Button(btn_frame, text="Save Comparisons",
                   command=self.save_comparisons, width=18).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Load Comparisons",
                   command=self.load_comparisons, width=18).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Clear All",
                   command=self.clear_comparisons, width=10).pack(side="left", padx=3)

    def open_comparison_browser(self):
        """
        Toplevel window listing every possible group pair as checkboxes.
        Already-selected comparisons are pre-ticked.
        User can check/uncheck, use quick-select buttons, then either
        replace the main list or add to it.
        """
        all_groups = []
        if self.animals_by_group:
            all_groups = list(self.animals_by_group.keys())
        elif self.off_food_groups:
            all_groups = list(self.off_food_groups.keys())

        if not all_groups:
            messagebox.showwarning("Warning",
                "No groups loaded yet.\nLoad a permutation package first.")
            return

        import itertools

        win = tk.Toplevel(self.root)
        win.title("All Possible Comparisons")
        win.geometry("720x600")
        win.resizable(True, True)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ttk.Frame(win, padding=(10, 8))
        hdr.pack(fill="x")
        n_pairs = len(all_groups) * (len(all_groups) - 1) // 2
        ttk.Label(hdr, text="Select comparisons",
                  font=("Helvetica", 12, "bold")).pack(side="left")
        ttk.Label(hdr,
                  text=f"  ({len(all_groups)} groups, {n_pairs} possible pairs)",
                  foreground="gray").pack(side="left")

        # ── Quick-select buttons ──────────────────────────────────────────────
        btn_row = ttk.Frame(win, padding=(10, 0))
        btn_row.pack(fill="x")
        check_vars = {}  # (grp_a, grp_b) -> BooleanVar

        def select_all():
            for v in check_vars.values(): v.set(True)
        def clear_all():
            for v in check_vars.values(): v.set(False)
        def invert():
            for v in check_vars.values(): v.set(not v.get())

        ttk.Button(btn_row, text="Select All", command=select_all, width=12).pack(side="left", padx=3, pady=4)
        ttk.Button(btn_row, text="Clear All",  command=clear_all,  width=12).pack(side="left", padx=3)
        ttk.Button(btn_row, text="Invert",     command=invert,     width=12).pack(side="left", padx=3)

        # ── Scrollable checkbox area ──────────────────────────────────────────
        outer = ttk.Frame(win)
        outer.pack(fill="both", expand=True, padx=10, pady=4)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        cw = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                   lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll( 1, "units"))

        # Pre-tick anything already in the main list
        already = {(a, b) for _, a, b in self.selected_comparisons}

        pairs = list(itertools.combinations(all_groups, 2))
        for i, (grp_a, grp_b) in enumerate(pairs):
            var = tk.BooleanVar(value=((grp_a, grp_b) in already or
                                       (grp_b, grp_a) in already))
            check_vars[(grp_a, grp_b)] = var
            bg = "#f0f0f0" if i % 2 == 0 else "white"
            row_f = tk.Frame(inner, background=bg)
            row_f.pack(fill="x")
            tk.Checkbutton(row_f, variable=var, background=bg).pack(side="left")
            tk.Label(row_f,
                     text=f"{grp_a}   vs   {grp_b}",
                     background=bg, font=("Courier", 9),
                     anchor="w").pack(side="left", fill="x", expand=True, padx=(0, 6))

        # ── Action buttons ────────────────────────────────────────────────────
        act = ttk.Frame(win, padding=(10, 8))
        act.pack(fill="x")

        def _apply(replace: bool):
            chosen = [(a, b) for (a, b), v in check_vars.items() if v.get()]
            if not chosen:
                messagebox.showwarning("Warning", "No comparisons selected.", parent=win)
                return
            if replace:
                self.selected_comparisons = []
                self.comp_listbox.delete(0, tk.END)
            added = 0
            for grp_a, grp_b in chosen:
                name = f"{grp_a} vs {grp_b}"
                if (name, grp_a, grp_b) in self.selected_comparisons:
                    continue
                self.selected_comparisons.append((name, grp_a, grp_b))
                self.comp_listbox.insert(tk.END, name)
                added += 1
            action = "Replaced list with" if replace else "Added"
            self.log(f"✓ {action} {added} comparisons from browser")
            win.destroy()

        ttk.Button(act, text="Replace List with Selected",
                   command=lambda: _apply(replace=True),
                   width=26).pack(side="left", padx=5)
        ttk.Button(act, text="Add Selected to List",
                   command=lambda: _apply(replace=False),
                   width=22).pack(side="left", padx=5)
        ttk.Button(act, text="Cancel",
                   command=win.destroy,
                   width=10).pack(side="right", padx=5)

    def save_comparisons(self):
        """Save current comparison list to a JSON file for reuse."""
        if not self.selected_comparisons:
            messagebox.showwarning("Warning", "No comparisons to save.")
            return

        fp = filedialog.asksaveasfilename(
            title="Save comparison list",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not fp:
            return

        try:
            data = [{"name": name, "group_a": a, "group_b": b}
                    for name, a, b in self.selected_comparisons]
            with open(fp, "w") as f:
                json.dump(data, f, indent=2)
            self.log(f"✓ Saved {len(data)} comparisons to {fp}")
            messagebox.showinfo("Saved", f"Saved {len(data)} comparisons to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save comparisons:\n{str(e)}")

    def load_comparisons(self):
        """Load a previously saved comparison list from JSON.
        
        Appends to any existing comparisons so you can build a combined
        family list by loading multiple files one after another.
        """
        fp = filedialog.askopenfilename(
            title="Load comparison list",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not fp:
            return

        try:
            with open(fp, "r") as f:
                data = json.load(f)

            added = 0
            skipped = 0
            for entry in data:
                name  = entry.get("name", "")
                grp_a = entry.get("group_a", "")
                grp_b = entry.get("group_b", "")

                if not grp_a or not grp_b:
                    skipped += 1
                    continue

                # Auto-generate name if missing
                if not name:
                    name = f"{grp_a} vs {grp_b}"

                # Skip exact duplicates already in the list
                if (name, grp_a, grp_b) in self.selected_comparisons:
                    skipped += 1
                    continue

                self.selected_comparisons.append((name, grp_a, grp_b))
                self.comp_listbox.insert(tk.END, name)
                added += 1

            msg = f"Loaded {added} comparisons"
            if skipped:
                msg += f" ({skipped} skipped — duplicates or missing groups)"
            self.log(f"✓ {msg} from {fp}")
            messagebox.showinfo("Loaded", msg)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load comparisons:\n{str(e)}")

    def clear_comparisons(self):
        """Remove all comparisons from the list."""
        self.selected_comparisons = []
        self.comp_listbox.delete(0, tk.END)
        self.log("✓ Comparisons cleared")

    def create_action_section(self, parent):
        frame = ttk.Frame(parent)
        frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Run Tests", command=self.run_tests, width=20).pack(side="left", padx=5)
        ttk.Button(frame, text="Run One-Sample Tests", command=self.run_one_sample_tests, width=20).pack(side="left", padx=5)
        ttk.Button(frame, text="Clear Results", command=self.clear_results, width=15).pack(side="left", padx=5)
        ttk.Button(frame, text="Save Results CSV", command=self.save_results, width=15).pack(side="left", padx=5)
        ttk.Button(frame, text="Save One-Sample CSV", command=self.save_one_sample_results, width=18).pack(side="left", padx=5)

    def create_results_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Results", padding="10")
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self.results_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Courier", 9)
        )
        self.results_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.results_text.yview)

    # Event handlers

    def load_permutation_package(self):
        """Load unified permutation package directory created by Richards script."""
        dirname = filedialog.askdirectory(
            title="Select permutation_data directory"
        )
        if not dirname:
            return
        
        pkg_dir = Path(dirname)
        
        try:
            self.log("\n" + "="*80)
            self.log("LOADING PERMUTATION PACKAGE")
            self.log("="*80)
            
            # 1. Load combined bootstrap results
            bootstrap_path = pkg_dir / "bootstrap_results_combined.csv"
            if bootstrap_path.exists():
                self.bootstrap_combined_df = pd.read_csv(bootstrap_path)
                
                n_on = len(self.bootstrap_combined_df[self.bootstrap_combined_df['data_type'] == 'on_food'])
                n_off = len(self.bootstrap_combined_df[self.bootstrap_combined_df['data_type'] == 'off_food'])
                
                self.log(f"\n✓ Loaded bootstrap results: {bootstrap_path.name}")
                self.log(f"  ON-food values: {n_on}")
                self.log(f"  OFF-food values: {n_off}")
                
                on_params = self.bootstrap_combined_df[
                    self.bootstrap_combined_df['data_type'] == 'on_food'
                ]['parameter'].unique()
                self.log(f"  ON-food parameters: {', '.join(on_params)}")
                
                if n_off > 0:
                    off_params = self.bootstrap_combined_df[
                        self.bootstrap_combined_df['data_type'] == 'off_food'
                    ]['parameter'].unique()
                    self.log(f"  OFF-food parameters: {', '.join(off_params)}")
                
                self.bootstrap_label.config(text="Package loaded", foreground="black")
            else:
                raise FileNotFoundError("bootstrap_results_combined.csv not found in package")
            
            # Compute ON-food y_initial group means for food_encounter_score denominator
            # This must match the Richards script: y_final / mean(y_initial for GROUP)
            if self.bootstrap_combined_df is not None:
                self.compute_on_food_y_initial_group_means(self.bootstrap_combined_df)
            
            # 2. Load ON-food windowed data
            windowed_path = pkg_dir / "on_food_windowed.csv"
            if windowed_path.exists():
                self.groups = organize_animals_by_group(windowed_path)
                self.animals_by_group = {label: info['animals'] for label, info in self.groups.items()}
                
                n_groups = len(self.groups)
                total_animals = sum(info['n_animals'] for info in self.groups.values())
                
                self.log(f"\n✓ Loaded ON-food windowed data: {windowed_path.name}")
                self.log(f"  Groups: {n_groups}, Total animals: {total_animals}")
                
                self.animal_label.config(text="Package loaded", foreground="black")
            else:
                self.log("\n⚠ Warning: on_food_windowed.csv not found (OFF-food testing only)")
            
            # 3. Load OFF-food individual baselines
            # These are used for the baseline_speed permutation test (no curve fitting).
            off_path = pkg_dir / "off_food_individual_baselines.csv"
            if off_path.exists():
                self.off_food_individual_df = pd.read_csv(off_path)
                self.off_food_groups = organize_off_food_baselines_by_group_from_df(self.off_food_individual_df)

                n_groups = len(self.off_food_groups)
                total_animals = sum(info['n_animals'] for info in self.off_food_groups.values())

                self.log(f"\n✓ Loaded OFF-food individual baselines: {off_path.name}")
                self.log(f"  Groups: {n_groups}, Total animals: {total_animals}")

                self.off_food_label.config(text="Package loaded", foreground="black")
            else:
                self.log("\n⚠ Warning: off_food_individual_baselines.csv not found (ON-food testing only)")

            # 4. Compute food_detection_score denominator (off-food group mean).
            # Prefer the bootstrap baselines CSV so the denominator matches what the
            # Richards script used when computing the observed food_detection_score
            # (Richards takes mean of 10,000 bootstrap replicate means, not raw
            # individual means, so we must use the same source here).
            # Fall back to individual baselines if the bootstrap file is absent.
            bootstrap_off_path = pkg_dir / "off_food_bootstrap_baselines.csv"
            if bootstrap_off_path.exists():
                bootstrap_off_df = pd.read_csv(bootstrap_off_path)
                self.compute_off_food_group_means_from_df(bootstrap_off_df)
                self.log(f"\n✓ Using OFF-food bootstrap baselines for food_detection_score denominator")
                self.log(f"  (matches Richards script: mean of bootstrap replicate means per group)")
            elif self.off_food_individual_df is not None:
                self.compute_off_food_group_means_from_df(self.off_food_individual_df)
                self.log(f"\n⚠ off_food_bootstrap_baselines.csv not found — falling back to individual baselines")
                self.log(f"  for food_detection_score denominator (minor numerical difference)")
            else:
                self.log(f"\n⚠ No OFF-food data available — food_detection_score cannot be tested")
            
            self.permutation_package_dir = pkg_dir
            
            # UPDATE COMPARISON DROPDOWNS with available groups
            if self.groups is not None:
                group_names = list(self.groups.keys())
                self.group_a_combo['values'] = group_names
                self.group_b_combo['values'] = group_names
                
                n_groups = len(self.groups)
                n_animals = sum(info['n_animals'] for info in self.groups.values())
                self.groups_display.config(
                    text=f"{n_groups} ON-food groups, {n_animals} animals total",
                    foreground="black"
                )
                
                self.log(f"\n✓ Updated comparison dropdowns: {n_groups} ON-food groups available")
            
            if self.off_food_groups is not None:
                n_off_groups = len(self.off_food_groups)
                n_off_animals = sum(info['n_animals'] for info in self.off_food_groups.values())
                self.log(f"✓ OFF-food groups: {n_off_groups} groups, {n_off_animals} animals")
            
            self.log("\n" + "="*80)
            self.log("✓ PACKAGE LOADED SUCCESSFULLY!")
            self.log("="*80)
            
            messagebox.showinfo("Success", "Permutation package loaded!\n\nReady for permutation testing.")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            messagebox.showerror("Error", f"Failed to load package:\n{str(e)}")
            self.log(f"\n✗ ERROR: {error_msg}")
    
    def compute_off_food_group_means_from_df(self, off_food_df: pd.DataFrame):
        """Compute OFF-food group means from DataFrame for food_detection_score denominator."""
        if off_food_df is None or len(off_food_df) == 0:
            return
        
        try:
            uniq_t = off_food_df['treatment'].dropna().unique()
            use_treatment = (len(uniq_t) > 1)
        except Exception:
            use_treatment = False
        
        group_means = {}
        
        if use_treatment:
            for (treatment, sex, genotype), group_df in \
                off_food_df.groupby(['treatment', 'sex', 'strain_genotype']):
                baselines = group_df['baseline_speed'].dropna()
                if len(baselines) > 0:
                    key = (str(treatment), str(sex), str(genotype))
                    group_means[key] = float(np.mean(baselines))
        else:
            for (sex, genotype), group_df in \
                off_food_df.groupby(['sex', 'strain_genotype']):
                baselines = group_df['baseline_speed'].dropna()
                if len(baselines) > 0:
                    key = (str(sex), str(genotype))
                    group_means[key] = float(np.mean(baselines))
        
        self.off_food_group_mean = group_means
        
        mode = "treatment×sex×genotype" if use_treatment else "sex×genotype"
        self.log(f"\n✓ Computed OFF-food group means:")
        self.log(f"  Grouping: {mode}")
        for key, mean_val in group_means.items():
            self.log(f"  {key}: {mean_val:.4f} mm/s")

    def compute_on_food_y_initial_group_means(self, bootstrap_combined_df: pd.DataFrame):
        """
        Compute ON-food y_initial group means for use as the food_encounter_score denominator.

        This matches the Richards script definition:
            food_encounter_score = y_final / mean(y_initial for GROUP)

        The mean is computed from the ON-food bootstrap/individual replicate results,
        exactly as in add_behavioral_context() in the Richards script.

        Args:
            bootstrap_combined_df: Combined bootstrap results CSV with 'data_type', 'parameter',
                                   'value', 'treatment', 'sex', 'strain_genotype' columns.
        """
        if bootstrap_combined_df is None or len(bootstrap_combined_df) == 0:
            return

        try:
            on_food_df = bootstrap_combined_df[
                (bootstrap_combined_df['data_type'] == 'on_food') &
                (bootstrap_combined_df['parameter'] == 'y_initial')
            ].copy()

            if len(on_food_df) == 0:
                self.log("\n⚠ Warning: No ON-food y_initial values found in bootstrap results")
                self.log("   food_encounter_score will not be available")
                return

            # Determine grouping (with or without treatment column)
            try:
                uniq_t = on_food_df['treatment'].dropna().unique()
                use_treatment = (len(uniq_t) > 1)
            except KeyError:
                use_treatment = False

            group_means = {}

            if use_treatment:
                for (treatment, sex, genotype), group_df in \
                        on_food_df.groupby(['treatment', 'sex', 'strain_genotype']):
                    vals = group_df['value'].dropna()
                    if len(vals) > 0:
                        key = (str(treatment), str(sex), str(genotype))
                        group_means[key] = float(np.mean(vals))
            else:
                for (sex, genotype), group_df in \
                        on_food_df.groupby(['sex', 'strain_genotype']):
                    vals = group_df['value'].dropna()
                    if len(vals) > 0:
                        key = (str(sex), str(genotype))
                        group_means[key] = float(np.mean(vals))

            self.on_food_y_initial_group_mean = group_means

            mode = "treatment×sex×genotype" if use_treatment else "sex×genotype"
            self.log(f"\n✓ Computed ON-food y_initial group means (food_encounter_score denominator):")
            self.log(f"  Grouping: {mode}")
            for key, mean_val in group_means.items():
                self.log(f"  {key}: {mean_val:.4f} mm/s")

        except Exception as e:
            self.log(f"\n⚠ Warning: Could not compute ON-food y_initial group means: {e}")

    
    def load_bootstrap(self):
        filepath = filedialog.askopenfilename(
            title="Select Bootstrap Results CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.bootstrap_df = load_bootstrap_results(Path(filepath), self.parameter.get())
                self.bootstrap_label.config(text=Path(filepath).name, foreground="black")
                # Try to auto-load trajectory aggregation choice from Richards output folder
                try:
                    settings_path = Path(filepath).parent / "run_settings.json"
                    if settings_path.exists():
                        with open(settings_path, 'r', encoding='utf-8') as f:
                            settings = json.load(f)
                        agg = str(settings.get('trajectory_aggregation', '')).strip().lower()
                        if agg in ('median', 'mean'):
                            self.trajectory_agg.set(agg)
                            self.log(f"  Detected trajectory aggregation from run_settings.json: {agg}")
                except Exception:
                    pass
                self.log(f"✓ Loaded bootstrap results: {len(self.bootstrap_df)} replicates")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load bootstrap results:\n{str(e)}")

    def load_animals(self):
        filepath = filedialog.askopenfilename(
            title="Select Windowed Speed Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.groups = organize_animals_by_group(Path(filepath))
                self.animals_by_group = {label: info['animals'] for label, info in self.groups.items()}

                self.animal_label.config(text=Path(filepath).name, foreground="black")

                group_names = list(self.groups.keys())
                self.group_a_combo['values'] = group_names
                self.group_b_combo['values'] = group_names

                n_groups = len(self.groups)
                n_animals = sum(info['n_animals'] for info in self.groups.values())
                self.groups_display.config(text=f"{n_groups} groups, {n_animals} animals total",
                                           foreground="black")

                self.log(f"✓ Loaded animal data: {n_groups} groups, {n_animals} animals")
                # If OFF-food baselines already loaded, compute group means now
                self.compute_off_food_group_means()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load animal data:\n{str(e)}")

    def load_off_food(self):
        filepath = filedialog.askopenfilename(
            title="Select OFF-food Baseline CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.off_food_df = pd.read_csv(filepath)

                # Validate required columns (group columns needed for proper group mean calculation)
                required_cols = ['assay_num', 'track_num', 'baseline_speed', 'treatment', 'sex', 'strain_genotype']
                missing = [col for col in required_cols if col not in self.off_food_df.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}\n\n"
                                   f"OFF-food CSV must include group columns (treatment, sex, strain_genotype)")

                # Create lookup dict: (assay, track) -> baseline_speed
                # NOTE: This is kept for legacy compatibility but group means are calculated
                # directly from OFF-food data WITHOUT requiring ID matching to ON-food
                self.off_food_by_animal = {}
                for _, row in self.off_food_df.iterrows():
                    key = (int(row['assay_num']), int(row['track_num']))
                    self.off_food_by_animal[key] = row['baseline_speed']

                self.off_food_label.config(text=Path(filepath).name, foreground="black")

                n_animals = len(self.off_food_by_animal)
                self.log(f"✓ Loaded OFF-food baselines: {n_animals} animals")
                self.log(f"  Mean baseline: {np.mean(list(self.off_food_by_animal.values())):.4f} mm/s")
                self.compute_off_food_group_means()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load OFF-food data:\n{str(e)}")
    
    def compute_off_food_group_means(self):
        """Compute mean OFF-food baseline per GROUP from OFF-food data ONLY.
        
        CRITICAL FIX: This now calculates group means directly from the OFF-food CSV
        WITHOUT attempting to match animal IDs to ON-food data. OFF-food and ON-food
        experiments are SEPARATE with independent assay/track numbering.
        
        Grouping logic:
          - If OFF-food has only ONE treatment level, baselines are grouped by (sex, strain_genotype)
            and applied to all treatments in ON-food data.
          - Otherwise, baselines are grouped by (treatment, sex, strain_genotype).
        """
        if self.off_food_df is None or len(self.off_food_df) == 0:
            self.log("✗ No OFF-food data loaded")
            return
        
        # Decide grouping mode based on OFF-food file content
        use_treatment = False
        try:
            uniq_t = self.off_food_df['treatment'].dropna().unique()
            use_treatment = (len(uniq_t) > 1)
        except Exception:
            use_treatment = False
        
        group_means = {}
        
        # Group OFF-food data directly (no reference to ON-food animals)
        if use_treatment:
            # Group by (treatment, sex, genotype)
            for (treatment, sex, genotype), group_df in self.off_food_df.groupby(['treatment', 'sex', 'strain_genotype']):
                baselines = group_df['baseline_speed'].dropna()
                if len(baselines) > 0:
                    key = (str(treatment), str(sex), str(genotype))
                    group_means[key] = float(np.mean(baselines))
        else:
            # Group by (sex, genotype) only - ignore treatment
            for (sex, genotype), group_df in self.off_food_df.groupby(['sex', 'strain_genotype']):
                baselines = group_df['baseline_speed'].dropna()
                if len(baselines) > 0:
                    key = (str(sex), str(genotype))
                    group_means[key] = float(np.mean(baselines))
        
        self.off_food_group_mean = group_means
        
        mode = "treatment×sex×genotype" if use_treatment else "sex×genotype (treatment ignored)"
        self.log(f"✓ Computed OFF-food GROUP mean baselines: {len(group_means)} groups ({mode})")
        
        # Report group sizes from OFF-food data
        for key, mean_val in group_means.items():
            if use_treatment:
                mask = (self.off_food_df['treatment'] == key[0]) & \
                       (self.off_food_df['sex'] == key[1]) & \
                       (self.off_food_df['strain_genotype'] == key[2])
            else:
                mask = (self.off_food_df['sex'] == key[0]) & \
                       (self.off_food_df['strain_genotype'] == key[1])
            n = mask.sum()
            self.log(f"  {key}: n={n} animals, mean={mean_val:.4f} mm/s")


    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.output_label.config(text=str(self.output_dir), foreground="black")
            self.log(f"✓ Output directory: {self.output_dir}")

    def add_comparison(self):
        group_a = self.group_a_var.get()
        group_b = self.group_b_var.get()
        name = self.comp_name_var.get()

        if not group_a or not group_b:
            messagebox.showwarning("Warning", "Please select both groups")
            return

        if not name:
            name = f"{group_a} vs {group_b}"

        for comp in self.selected_comparisons:
            if comp[1] == group_a and comp[2] == group_b:
                messagebox.showwarning("Warning", "This comparison is already added")
                return

        self.selected_comparisons.append((name, group_a, group_b))
        self.comp_listbox.insert(tk.END, f"{name}")
        self.log(f"✓ Added comparison: {name}")

        self.comp_name_var.set("")

    def remove_comparison(self):
        selection = self.comp_listbox.curselection()
        if selection:
            idx = selection[0]
            self.comp_listbox.delete(idx)
            removed = self.selected_comparisons.pop(idx)
            self.log(f"✗ Removed comparison: {removed[0]}")

    def run_tests(self):
        # Determine data source: package vs individual files
        using_package = (self.bootstrap_combined_df is not None)
        
        # Basic validation
        if not using_package and self.bootstrap_df is None:
            messagebox.showwarning("Warning", "Please load bootstrap results or package first")
            return

        if not self.selected_comparisons:
            messagebox.showwarning("Warning", "Please add at least one comparison")
            return

        try:
            param = self.parameter.get()
            
            # Determine if testing OFF-food or ON-food
            is_off_food = (param == 'baseline_speed')
            data_type = 'off_food' if is_off_food else 'on_food'
            
            self.log("\n" + "=" * 80)
            self.log(f"RUNNING PERMUTATION TESTS - {data_type.upper()}")
            self.log("=" * 80)
            
            # Validate data availability for selected parameter
            if is_off_food:
                if not using_package and self.off_food_groups is None:
                    messagebox.showerror("Error", 
                        "OFF-food baseline data not loaded.\n"
                        "Load a permutation package with OFF-food data.")
                    return
                if using_package and self.off_food_groups is None:
                    messagebox.showerror("Error",
                        "Package does not contain OFF-food data.\n"
                        "Load package with off_food_individual_baselines.csv")
                    return
                animals_by_group = self.off_food_groups
            else:
                if not using_package and self.animals_by_group is None:
                    messagebox.showwarning("Warning", "Please load animal data first")
                    return
                if using_package and self.animals_by_group is None:
                    messagebox.showerror("Error",
                        "Package does not contain ON-food data.\n"
                        "Load package with on_food_windowed.csv")
                    return
                animals_by_group = self.animals_by_group

            # Get observed differences
            if using_package:
                self.log(f"\nGetting observed differences from package ({data_type})...")
                self.comparisons = get_observed_differences_unified(
                    bootstrap_df=self.bootstrap_combined_df,
                    comparisons=self.selected_comparisons,
                    parameter=param,
                    data_type=data_type
                )
            else:
                self.log(f"\nGetting observed differences from bootstrap file...")
                self.comparisons = get_observed_differences_from_bootstrap(
                    self.bootstrap_df,
                    self.selected_comparisons,
                    param
                )

            if not self.comparisons:
                messagebox.showerror("Error", "No valid comparisons found. Check group labels.")
                return

            # Special handling for behavioral context scores
            if param == 'food_detection_score':
                if self.off_food_group_mean is None:
                    messagebox.showerror("Error",
                        "food_detection_score requires OFF-food baselines.\n"
                        "Load OFF-food data with package or individual file.")
                    return
                self.log(f"\n✓ Testing food_detection_score")
                self.log(f"   Calculated as: y_initial / mean(OFF-food baseline for GROUP)")
                
            elif param == 'food_encounter_score':
                if self.on_food_y_initial_group_mean is None:
                    messagebox.showerror("Error",
                        "food_encounter_score requires ON-food bootstrap results.\n"
                        "Load a permutation package (bootstrap_results_combined.csv must be present).")
                    return
                self.log(f"\n✓ Testing food_encounter_score")
                self.log(f"   Permutation statistic: y_final / y_initial from same aggregated fit")
                self.log(f"   (Fixed group-mean denominator would bias null when groups differ in y_initial)")

            self.log(f"\n✓ Found {len(self.comparisons)} valid comparisons")
            for comp in self.comparisons:
                self.log(f"  {comp.name}: {comp.observed_diff:.4f}")

            # Progress callback
            def update_progress(current, total):
                self.log(f"  Progress: {current}/{total} ({100 * current // total}%)")
                self.root.update()

            # Run tests
            method = self.test_method.get()
            n_cores = self.n_cores.get()

            # Choose permutation function based on parameter type
            if is_off_food:
                # OFF-food baseline permutation (no curve fitting)
                if method == 'maxt':
                    self.log("\n✓ Running Max-T permutation test on OFF-food baselines...")
                    self.log(f"  Settings: {self.n_permutations.get()} permutations")
                    self.log(f"  Method: CLASSICAL PERMUTATION on baseline speeds")
                    self.log(f"  Using {n_cores} of {self.max_cores} CPU cores")
                    self.results = maxt_permutation_test_baseline(
                        comparisons=self.comparisons,
                        animals_by_group=animals_by_group,
                        n_permutations=self.n_permutations.get(),
                        n_cores=n_cores,
                        progress_callback=update_progress
                    )
                else:
                    self.log("\n✓ Running standard permutation tests on OFF-food baselines...")
                    self.log(f"  Settings: {self.n_permutations.get()} permutations")
                    self.log(f"  Method: CLASSICAL PERMUTATION on baseline speeds")
                    self.log(f"  Using {n_cores} of {self.max_cores} CPU cores")
                    self.results = []
                    for i, comp in enumerate(self.comparisons):
                        self.log(f"\n  Test {i + 1}/{len(self.comparisons)}: {comp.name}")
                        result = standard_permutation_test_baseline(
                            group_a_animals=animals_by_group[comp.group_a],
                            group_b_animals=animals_by_group[comp.group_b],
                            observed_diff=comp.observed_diff,
                            n_permutations=self.n_permutations.get(),
                            n_cores=n_cores,
                            progress_callback=update_progress
                        )
                        self.results.append(result)
            else:
                # ON-food Richards parameter permutation (with curve fitting)
                if method == 'maxt':
                    self.log("\n✓ Running Max-T permutation test...")
                    self.log(f"  Settings: {self.n_permutations.get()} permutations")
                    self.log(f"  Method: CLASSICAL PERMUTATION (no bootstrap within permutation)")
                    self.log(f"  Each permutation aggregates ALL animals per group, fits 1 curve per group")
                    self.log(f"  Using {n_cores} of {self.max_cores} CPU cores")
                    self.results = maxt_permutation_test(
                        comparisons=self.comparisons,
                        animals_by_group=animals_by_group,
                        parameter=param,
                        n_permutations=self.n_permutations.get(),
                        n_cores=n_cores,
                        trajectory_agg=self.trajectory_agg.get(),
                        off_food_group_mean=self.off_food_group_mean,
                        on_food_y_initial_group_mean=self.on_food_y_initial_group_mean,
                        progress_callback=update_progress
                    )
                else:
                    self.log("\n✓ Running standard permutation tests...")
                    self.log(f"  Settings: {self.n_permutations.get()} permutations")
                    self.log(f"  Method: CLASSICAL PERMUTATION (no bootstrap within permutation)")
                    self.log(f"  Each permutation aggregates ALL animals per group, fits 1 curve per group")
                    self.log(f"  Using {n_cores} of {self.max_cores} CPU cores")
                    self.results = []
                    for i, comp in enumerate(self.comparisons):
                        self.log(f"\n  Test {i + 1}/{len(self.comparisons)}: {comp.name}")
                        result = standard_permutation_test(
                            group_a_animals=animals_by_group[comp.group_a],
                            group_b_animals=animals_by_group[comp.group_b],
                            observed_diff=comp.observed_diff,
                            parameter=param,
                            n_permutations=self.n_permutations.get(),
                            n_cores=n_cores,
                            trajectory_agg=self.trajectory_agg.get(),
                            off_food_group_mean=self.off_food_group_mean,
                            on_food_y_initial_group_mean=self.on_food_y_initial_group_mean,
                            progress_callback=update_progress
                        )
                        self.results.append(result)

            # Display results
            self.display_results()

            self.log("\n" + "=" * 80)
            self.log("✓ TESTS COMPLETE!")
            self.log("=" * 80)

            messagebox.showinfo("Success", "Tests completed successfully!")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            messagebox.showerror("Error", f"Failed to run tests:\n{str(e)}")
            self.log(f"\n✗ ERROR: {error_msg}")

    def display_results(self):
        # Get p-values
        p_values = np.array([r.p_value for r in self.results])

        # Apply correction if needed
        correction = None if self.test_method.get() == 'maxt' else self.correction_method.get()
        if correction and correction != 'none':
            if correction == 'bonferroni':
                p_adj = bonferroni_correction(p_values)
                correction_name = "Bonferroni"
            elif correction == 'holm':
                p_adj = holm_bonferroni_correction(p_values)
                correction_name = "Holm-Bonferroni"
            elif correction == 'fdr':
                p_adj = fdr_benjamini_hochberg(p_values)
                correction_name = "FDR (B-H)"
            else:
                p_adj = p_values
                correction_name = "None"
        else:
            p_adj = p_values
            correction_name = "Max-T" if self.test_method.get() == 'maxt' else "None"

        # Format results
        alpha = self.alpha.get()
        output = io.StringIO()

        output.write("\n" + "=" * 100 + "\n")
        output.write("PERMUTATION TEST RESULTS\n")
        output.write(f"Method: {self.results[0].method.upper()}\n")
        output.write(f"Correction: {correction_name}\n")
        output.write(f"Significance level: α = {alpha}\n")
        output.write("=" * 100 + "\n\n")

        output.write(f"{'Comparison':<40} {'Group A':<12} {'Group B':<12} {'Diff':>10} {'p-value':>10} {'p-adj':>10} {'Sig':>5}\n")
        output.write("-" * 100 + "\n")

        for comp, result, p_raw, p_a in zip(self.comparisons, self.results, p_values, p_adj):
            sig = "*" if p_a < alpha else "ns"
            output.write(
                f"{comp.name:<40} {comp.group_a_mean:>12.4f} {comp.group_b_mean:>12.4f} "
                f"{comp.observed_diff:>10.4f} {p_raw:>10.4f} {p_a:>10.4f} {sig:>5}\n"
            )

        output.write("=" * 100 + "\n")
        output.write(f"\nSignificant comparisons: {sum(p_adj < alpha)} / {len(self.comparisons)}\n")

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, output.getvalue())

    def clear_results(self):
        self.results = []
        self.comparisons = []
        self.one_sample_results = []
        self.results_text.delete(1.0, tk.END)
        self.log("✓ Results cleared")

    def save_results(self):
        if not self.output_dir:
            messagebox.showwarning("Warning", "Please select output directory first")
            return

        if not self.results:
            messagebox.showwarning("Warning", "No results to save")
            return

        try:
            p_values = np.array([r.p_value for r in self.results])

            # Allow post-hoc correction on any method, including Max-T.
            # Max-T already controls FWER; applying BH FDR on top trades FWER
            # control for more power (appropriate when testing many comparisons).
            correction = self.correction_method.get()
            if correction and correction != 'none':
                if correction == 'bonferroni':
                    p_adj = bonferroni_correction(p_values)
                elif correction == 'holm':
                    p_adj = holm_bonferroni_correction(p_values)
                elif correction == 'fdr':
                    p_adj = fdr_benjamini_hochberg(p_values)
                else:
                    p_adj = p_values
            else:
                p_adj = p_values

            results_df = pd.DataFrame({
                'comparison': [c.name for c in self.comparisons],
                'group_a': [c.group_a for c in self.comparisons],
                'group_b': [c.group_b for c in self.comparisons],
                'group_a_mean': [c.group_a_mean for c in self.comparisons],
                'group_b_mean': [c.group_b_mean for c in self.comparisons],
                'observed_diff': [c.observed_diff for c in self.comparisons],
                'p_value': p_values,
                'p_adjusted': p_adj,
                'significant': p_adj < self.alpha.get(),
                'method': [r.method for r in self.results],
                'n_permutations': [r.n_permutations for r in self.results]
            })

            output_path = self.output_dir / "permutation_results.csv"
            results_df.to_csv(output_path, index=False)

            self.log(f"\n✓ Results saved to {output_path}")
            messagebox.showinfo("Success", f"Results saved to:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")

    def run_one_sample_tests(self):
        """
        Run one-sample bootstrap CI tests for detection and encounter scores vs mu0.

        For each group, tests whether the bootstrap distribution of the score
        excludes mu0 at the 95% CI level (2.5/97.5 percentiles).  No permutation
        needed — the bootstrap distribution IS the sampling distribution.
        """
        if self.bootstrap_combined_df is None:
            messagebox.showwarning("Warning",
                "No bootstrap data loaded.\n"
                "Load a permutation package first.")
            return

        try:
            mu0 = float(self.mu0_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Error", "Invalid μ₀ value.")
            return

        test_params = []
        on_df = self.bootstrap_combined_df[
            self.bootstrap_combined_df['data_type'] == 'on_food'
        ]
        available = on_df['parameter'].unique().tolist()
        for p in ['food_detection_score', 'food_encounter_score']:
            if p in available:
                test_params.append(p)

        if not test_params:
            messagebox.showwarning("Warning",
                "Neither food_detection_score nor food_encounter_score found in\n"
                "bootstrap_results_combined.csv.\n\n"
                "Re-run Richards analysis with OFF-food data loaded to generate these scores.")
            return

        self.one_sample_results = []
        alpha = self.alpha.get()

        self.log("\n" + "=" * 70)
        self.log(f"ONE-SAMPLE BOOTSTRAP CI TESTS  (μ₀ = {mu0})")
        self.log("=" * 70)
        self.log(f"Primary readout: does μ₀ fall outside the 95% CI?")
        self.log(f"  CI = [2.5th percentile, 97.5th percentile] of bootstrap distribution")
        self.log(f"  p-value = empirical two-tailed tail probability at μ₀, floor = 1/(n+1)\n")

        group_cols = ['treatment', 'sex', 'strain_genotype']

        for param in test_params:
            param_df = on_df[on_df['parameter'] == param]

            self.log(f"─── {param} ───")
            self.log(f"{'Group':<40} {'Mean':>8} {'CI_lo':>8} {'CI_hi':>8} "
                     f"{'Diff':>8} {'p':>8} {'Sig':>5}")
            self.log("-" * 90)

            param_results = []
            for cols in [['treatment', 'sex', 'strain_genotype'],
                         ['sex', 'strain_genotype']]:
                # Use 3-col grouping if treatment has >1 unique value
                try:
                    groups = param_df.groupby(cols)
                    if len(groups) > 0:
                        break
                except KeyError:
                    continue

            for group_key, group_df in param_df.groupby(group_cols,
                                                         observed=True,
                                                         dropna=True):
                scores = group_df['value'].dropna().values
                if len(scores) < 10:
                    self.log(f"  {str(group_key):<38} SKIPPED (n={len(scores)} < 10)")
                    continue

                label = "|".join(str(k) for k in group_key)
                result = bootstrap_one_sample_test(scores, label, param, mu0)
                self.one_sample_results.append(result)
                param_results.append(result)

                sig_str = "***" if result.p_value < 0.001 else \
                          "**"  if result.p_value < 0.01  else \
                          "*"   if result.p_value < alpha  else "ns"
                direction = "↑" if result.mean > mu0 else "↓"
                sig_marker = f"{sig_str}{direction}" if result.significant else sig_str

                self.log(f"  {label:<38} {result.mean:>8.4f} {result.ci_lo:>8.4f} "
                         f"{result.ci_hi:>8.4f} {result.mean_minus_mu0:>+8.4f} "
                         f"{result.p_value:>8.4f} {sig_marker:>5}")

            # Apply BH FDR across groups for this parameter if selected
            correction = self.correction_method.get()
            if correction == 'fdr' and len(param_results) > 1:
                raw_p = np.array([r.p_value for r in param_results])
                adj_p = fdr_benjamini_hochberg(raw_p)
                self.log(f"\n  BH FDR adjusted p-values for {param}:")
                for r, ap in zip(param_results, adj_p):
                    sig_adj = ap < alpha
                    self.log(f"    {r.group:<38} raw p={r.p_value:.4f}  adj p={ap:.4f}"
                             f"  {'*' if sig_adj else 'ns'}")
            self.log("")

        self.log(f"✓ One-sample tests complete. "
                 f"n={len(self.one_sample_results)} groups tested.")
        self.log(f"  Use 'Save One-Sample CSV' to export results.\n")

    def save_one_sample_results(self):
        """Save one-sample CI test results to CSV."""
        if not self.one_sample_results:
            messagebox.showwarning("Warning", "No one-sample results to save.\n"
                                              "Run 'Run One-Sample Tests' first.")
            return

        if not self.output_dir:
            messagebox.showwarning("Warning", "Please select output directory first.")
            return

        try:
            rows = []
            for r in self.one_sample_results:
                rows.append({
                    'group':            r.group,
                    'parameter':        r.parameter,
                    'mu0':              r.mu0,
                    'mean':             r.mean,
                    'ci_lo_2.5pct':     r.ci_lo,
                    'ci_hi_97.5pct':    r.ci_hi,
                    'mean_minus_mu0':   r.mean_minus_mu0,
                    'p_value':          r.p_value,
                    'significant':      r.significant,
                    'n_bootstrap':      r.n_bootstrap,
                })
            df = pd.DataFrame(rows)
            out_path = self.output_dir / "one_sample_results.csv"
            df.to_csv(out_path, index=False)
            self.log(f"✓ One-sample results saved to {out_path}")
            messagebox.showinfo("Saved", f"One-sample results saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{str(e)}")

    def log(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    root = tk.Tk()
    app = PermutationTestingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
