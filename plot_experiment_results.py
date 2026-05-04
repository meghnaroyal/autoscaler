#!/usr/bin/env python3
"""
plot_experiment_results.py
--------------------------
Read the experiment CSVs produced by collect_experiment.py and generate
publication-quality figures plus a summary statistics file that can be copied
directly into a research paper.

Multi-run usage (recommended for academic credibility)
------------------------------------------------------
  # Run the experiment 3–5 times per scenario and collect numbered files:
  python collect_experiment.py --scenario gru --run 1
  python collect_experiment.py --scenario hpa --run 1
  # … repeat for --run 2, --run 3, …

  python plot_experiment_results.py
  # → automatically detects all run files and reports mean ± std

Single-run usage (quick check)
-------------------------------
  python collect_experiment.py --scenario gru
  python collect_experiment.py --scenario hpa
  python plot_experiment_results.py

Outputs written to results/
---------------------------
  replica_timeline.png        — Figure: replica count over time, GRU vs HPA
  cpu_timeline.png            — Figure: cluster CPU usage over time, GRU vs HPA
  scaling_lag.png             — Figure: scale-up and scale-down lag bar chart
  over_provisioning.png       — Figure: excess replica-seconds during recovery
  under_provisioning.png      — Figure: *** seconds where CPU/pod > SLA threshold (key!)
  resource_efficiency.png     — Figure: CPU-per-replica (utilisation density)
  paper_summary_figure.png    — Figure: 4×2 combined figure for the paper
  experiment_summary.txt      — Numeric results table for the paper

Burst event timeline (from kubernetes/k6-burst-test.js)
--------------------------------------------------------
  t=  0s  baseline        20 VUs
  t= 30s  spike-1 ramp   300 VUs
  t= 35s  spike-1 peak   300 VUs  ← key comparison window
  t= 95s  drop-1           5 VUs
  t=100s  recovery         20 VUs  ← over-provisioning window
  t=160s  spike-2 ramp   300 VUs
  t=165s  spike-2 peak   300 VUs
  t=225s  drop-2           5 VUs
  t=230s  cooldown         20 VUs
  t=260s  end
"""

import os
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"

# ── burst event timestamps (seconds from experiment start) ────────────────────

SPIKE1_START   =  30
SPIKE1_PEAK    =  35
SPIKE1_END     =  95
RECOVERY_END   = 160
SPIKE2_START   = 160
SPIKE2_PEAK    = 165
SPIKE2_END     = 225
COOLDOWN_START = 230
TEST_END       = 260

SPIKE_REGIONS  = [(SPIKE1_PEAK, SPIKE1_END), (SPIKE2_PEAK, SPIKE2_END)]

# ── under-provisioning threshold ─────────────────────────────────────────────
# Pod CPU limit is 1000 m (cpu-load-app.yaml).  When cpu/pod > 800 m the pod
# is using >80 % of its limit and is at risk of CPU throttling / high latency.
# This is the SLA-violation proxy used in the paper.

CPU_UNDERPROVISIONED_THRESHOLD_M = 800  # millicores

# ── plot style ────────────────────────────────────────────────────────────────

GRU_COLOR  = "#1f77b4"
HPA_COLOR  = "#d62728"
SPIKE_COLOR  = "#ffeeba"
ANNOT_COLOR  = "#555555"
GRID_ALPHA   = 0.35

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path):
    """Return a dict of np.arrays keyed by column name (NaN for missing cells)."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            data[col] = []
        for row in reader:
            for col in reader.fieldnames:
                v = row[col]
                try:
                    data[col].append(float(v) if v != "" else np.nan)
                except ValueError:
                    data[col].append(np.nan)
    return {k: np.array(v) for k, v in data.items()}


def load_scenario_runs(scenario):
    """Return (list_of_dicts, n_runs) for the given scenario.

    Searches for numbered run files first:
      results/{scenario}_experiment_run1.csv, …, runN.csv
    Falls back to the single-run file:
      results/{scenario}_experiment.csv
    """
    run_files = sorted(RESULTS_DIR.glob(f"{scenario}_experiment_run*.csv"))
    if run_files:
        print(f"  {scenario.upper()}: found {len(run_files)} run file(s): "
              f"{[f.name for f in run_files]}")
        return [load_csv(f) for f in run_files], len(run_files)

    single = RESULTS_DIR / f"{scenario}_experiment.csv"
    if single.exists():
        print(f"  {scenario.upper()}: found single-run file: {single.name}")
        return [load_csv(single)], 1

    print(f"\nERROR: No experiment data found for scenario '{scenario}'.")
    print(f"  Expected: {RESULTS_DIR}/{scenario}_experiment_runN.csv  (multi-run)")
    print(f"       or:  {RESULTS_DIR}/{scenario}_experiment.csv        (single-run)")
    print(f"\n  To collect data run:")
    print(f"    python collect_experiment.py --scenario {scenario} [--run N]")
    sys.exit(1)


# ── analysis helpers ──────────────────────────────────────────────────────────

def _valid(arr):
    """Forward-fill NaN then replace remaining NaN with 0."""
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last = out[i]
        elif not np.isnan(last):
            out[i] = last
    return np.where(np.isnan(out), 0, out)


def _nanval(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def find_first_scale_up(t, replicas, spike_start, baseline_replicas):
    """Seconds from spike_start until replicas first exceed baseline."""
    for i in np.where(t >= spike_start)[0]:
        if replicas[i] > baseline_replicas:
            return float(t[i] - spike_start)
    return np.nan


def find_first_scale_down(t, replicas, drop_start, peak_replicas):
    """Seconds from drop_start until replicas first drop below peak."""
    for i in np.where(t >= drop_start)[0]:
        if replicas[i] < peak_replicas:
            return float(t[i] - drop_start)
    return np.nan


def excess_replica_seconds(t, replicas, window_start, window_end, min_replicas):
    """∫ max(replicas − min, 0) dt over the window (trapezoidal rule)."""
    mask = (t >= window_start) & (t <= window_end)
    tw, rw = t[mask], replicas[mask]
    if len(tw) < 2:
        return 0.0
    return float(np.trapezoid(np.maximum(rw - min_replicas, 0), tw))


def under_provisioning_seconds(t, cpu_per_pod, threshold_m,
                                window_start, window_end, interval_s=5):
    """Count seconds where cpu_per_pod > threshold in the window.

    Uses a simple sample-count approach (each sample ≈ interval_s seconds).
    NaN samples are excluded from both numerator and denominator.
    """
    mask = (t >= window_start) & (t <= window_end)
    cpp  = cpu_per_pod[mask]
    valid = ~np.isnan(cpp)
    if not valid.any():
        return np.nan
    over  = np.sum(cpp[valid] > threshold_m)
    return float(over * interval_s)


# ── per-run statistics ────────────────────────────────────────────────────────

def _stats_single(gru, hpa):
    """Compute all metrics for one pair of (gru_csv, hpa_csv) dicts."""
    t_g = _valid(gru["elapsed_s"])
    r_g = _valid(gru["replicas"])
    t_h = _valid(hpa["elapsed_s"])
    r_h = _valid(hpa["replicas"])

    def baseline_r(t, r, window=30):
        vals = r[(t <= window) & (r > 0)]
        return float(np.median(vals)) if len(vals) else 1.0

    br_g = baseline_r(t_g, r_g)
    br_h = baseline_r(t_h, r_h)

    def peak_r(t, r, s, e):
        mask = (t >= s) & (t <= e)
        return float(np.max(r[mask])) if mask.any() else np.nan

    gru_pk_s1 = peak_r(t_g, r_g, SPIKE1_START, SPIKE1_END)
    hpa_pk_s1 = peak_r(t_h, r_h, SPIKE1_START, SPIKE1_END)
    gru_pk_s2 = peak_r(t_g, r_g, SPIKE2_START, SPIKE2_END)
    hpa_pk_s2 = peak_r(t_h, r_h, SPIKE2_START, SPIKE2_END)

    # CPU per pod arrays (already in CSV; recompute as guard)
    c_g   = _valid(gru["cpu_total_m"])
    p_g   = _valid(gru["pods_measured"])
    c_h   = _valid(hpa["cpu_total_m"])
    p_h   = _valid(hpa["pods_measured"])
    cpp_g = np.where(p_g > 0, c_g / p_g, np.nan)
    cpp_h = np.where(p_h > 0, c_h / p_h, np.nan)

    thr = CPU_UNDERPROVISIONED_THRESHOLD_M
    ivl = 5  # poll interval seconds

    return {
        "gru_baseline_replicas":  br_g,
        "hpa_baseline_replicas":  br_h,

        "gru_peak_replicas_s1":   gru_pk_s1,
        "hpa_peak_replicas_s1":   hpa_pk_s1,
        "gru_peak_replicas_s2":   gru_pk_s2,
        "hpa_peak_replicas_s2":   hpa_pk_s2,

        "gru_scaleup_lag_s1":     find_first_scale_up(t_g, r_g, SPIKE1_START, br_g),
        "hpa_scaleup_lag_s1":     find_first_scale_up(t_h, r_h, SPIKE1_START, br_h),
        "gru_scaleup_lag_s2":     find_first_scale_up(t_g, r_g, SPIKE2_START, br_g),
        "hpa_scaleup_lag_s2":     find_first_scale_up(t_h, r_h, SPIKE2_START, br_h),

        "gru_scaledown_lag_d1":   find_first_scale_down(t_g, r_g, SPIKE1_END, gru_pk_s1),
        "hpa_scaledown_lag_d1":   find_first_scale_down(t_h, r_h, SPIKE1_END, hpa_pk_s1),
        "gru_scaledown_lag_d2":   find_first_scale_down(t_g, r_g, SPIKE2_END, gru_pk_s2),
        "hpa_scaledown_lag_d2":   find_first_scale_down(t_h, r_h, SPIKE2_END, hpa_pk_s2),

        "gru_excess_rs_recovery": excess_replica_seconds(t_g, r_g, SPIKE1_END, RECOVERY_END, 1.0),
        "hpa_excess_rs_recovery": excess_replica_seconds(t_h, r_h, SPIKE1_END, RECOVERY_END, 1.0),
        "gru_excess_rs_cooldown": excess_replica_seconds(t_g, r_g, SPIKE2_END, TEST_END,      1.0),
        "hpa_excess_rs_cooldown": excess_replica_seconds(t_h, r_h, SPIKE2_END, TEST_END,      1.0),
        "gru_excess_rs_total":    (excess_replica_seconds(t_g, r_g, SPIKE1_END, RECOVERY_END, 1.0)
                                   + excess_replica_seconds(t_g, r_g, SPIKE2_END, TEST_END, 1.0)),
        "hpa_excess_rs_total":    (excess_replica_seconds(t_h, r_h, SPIKE1_END, RECOVERY_END, 1.0)
                                   + excess_replica_seconds(t_h, r_h, SPIKE2_END, TEST_END, 1.0)),

        # ── under-provisioning (SLA proxy) ────────────────────────────────
        # seconds where cpu_per_pod > CPU_UNDERPROVISIONED_THRESHOLD_M
        "gru_underprov_s1":       under_provisioning_seconds(t_g, cpp_g, thr,
                                      SPIKE1_PEAK, SPIKE1_END, ivl),
        "hpa_underprov_s1":       under_provisioning_seconds(t_h, cpp_h, thr,
                                      SPIKE1_PEAK, SPIKE1_END, ivl),
        "gru_underprov_s2":       under_provisioning_seconds(t_g, cpp_g, thr,
                                      SPIKE2_PEAK, SPIKE2_END, ivl),
        "hpa_underprov_s2":       under_provisioning_seconds(t_h, cpp_h, thr,
                                      SPIKE2_PEAK, SPIKE2_END, ivl),
        "gru_underprov_total":    (under_provisioning_seconds(t_g, cpp_g, thr,
                                       SPIKE1_PEAK, SPIKE1_END, ivl)
                                   + under_provisioning_seconds(t_g, cpp_g, thr,
                                         SPIKE2_PEAK, SPIKE2_END, ivl)),
        "hpa_underprov_total":    (under_provisioning_seconds(t_h, cpp_h, thr,
                                       SPIKE1_PEAK, SPIKE1_END, ivl)
                                   + under_provisioning_seconds(t_h, cpp_h, thr,
                                         SPIKE2_PEAK, SPIKE2_END, ivl)),

        "gru_peak_cpu_per_pod_s1": float(np.nanmax(cpp_g[(t_g >= SPIKE1_START)
                                                          & (t_g <= SPIKE1_END)]))
                                   if ((t_g >= SPIKE1_START) & (t_g <= SPIKE1_END)).any()
                                   else np.nan,
        "hpa_peak_cpu_per_pod_s1": float(np.nanmax(cpp_h[(t_h >= SPIKE1_START)
                                                          & (t_h <= SPIKE1_END)]))
                                   if ((t_h >= SPIKE1_START) & (t_h <= SPIKE1_END)).any()
                                   else np.nan,
    }


def compute_stats_multi(runs_g, runs_h):
    """Aggregate stats over multiple experiment runs.

    Returns a dict with:
      {key}      — mean across runs  (NaN if all runs had NaN)
      {key}_std  — std across runs   (0.0 for a single run)
      n_runs     — number of runs used
    """
    per_run = [_stats_single(g, h) for g, h in zip(runs_g, runs_h)]
    result  = {"n_runs": len(runs_g)}
    for k in per_run[0]:
        vals  = [r[k] for r in per_run if not _nanval(r[k])]
        result[k]           = float(np.mean(vals))          if vals else np.nan
        result[k + "_std"]  = float(np.std(vals, ddof=0))   if len(vals) > 1 else 0.0
    return result


# ── shared annotation helper ──────────────────────────────────────────────────

def _annotate_bursts(ax, label_alpha=0.6):
    ylim  = ax.get_ylim()
    yspan = ylim[1] - ylim[0]
    for (s, e) in SPIKE_REGIONS:
        ax.axvspan(s, e, color=SPIKE_COLOR, alpha=0.45, zorder=0)
    for t_drop in [SPIKE1_END, SPIKE2_END]:
        ax.axvline(t_drop, color="#888888", ls="--", lw=1.0, alpha=0.7)
    for t_ev, label in [
        (SPIKE1_START, "Spike 1"), (SPIKE1_END, "Drop 1"),
        (SPIKE2_START, "Spike 2"), (SPIKE2_END, "Drop 2"),
    ]:
        ax.text(t_ev + 2, ylim[0] + yspan * 0.96, label,
                fontsize=8, color=ANNOT_COLOR, alpha=label_alpha, va="top", ha="left")


def _error_kw():
    return {"elinewidth": 1.4, "capsize": 4, "capthick": 1.4}


def _safe_err(stats, key):
    """Return std value for key; 0 if absent or NaN (safe for matplotlib yerr)."""
    v = stats.get(key + "_std", 0.0)
    return 0.0 if _nanval(v) else v


def _safe_val(stats, key):
    v = stats.get(key, np.nan)
    return np.nan if _nanval(v) else v


# ── individual figures ────────────────────────────────────────────────────────

def _best_single_run(runs):
    """Return the single run whose elapsed times are the longest (most data)."""
    return max(runs, key=lambda d: _valid(d["elapsed_s"]).max())


def fig_replica_timeline(runs_g, runs_h):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    g = _best_single_run(runs_g)
    h = _best_single_run(runs_h)
    t_g, r_g = _valid(g["elapsed_s"]), _valid(g["replicas"])
    t_h, r_h = _valid(h["elapsed_s"]), _valid(h["replicas"])

    ax.step(t_g, r_g, where="post", color=GRU_COLOR, lw=2.0, label="GRU autoscaler", zorder=3)
    ax.step(t_h, r_h, where="post", color=HPA_COLOR,  lw=2.0, label="HPA (reactive)",
            zorder=3, ls="--")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Replica count")
    ax.set_title("Replica Count Over Time — GRU vs HPA (k6 burst test)")
    ax.set_xlim(0, max(t_g.max(), t_h.max()) + 5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="upper right")
    _annotate_bursts(ax)
    fig.tight_layout()
    out = RESULTS_DIR / "replica_timeline.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_cpu_timeline(runs_g, runs_h):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    g = _best_single_run(runs_g)
    h = _best_single_run(runs_h)
    t_g, c_g = _valid(g["elapsed_s"]), g["cpu_total_m"].copy()
    t_h, c_h = _valid(h["elapsed_s"]), h["cpu_total_m"].copy()

    ax.plot(t_g, c_g, color=GRU_COLOR, lw=1.8, label="GRU — cluster CPU (m)")
    ax.plot(t_h, c_h, color=HPA_COLOR,  lw=1.8, label="HPA — cluster CPU (m)", ls="--")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Total cluster CPU (millicores)")
    ax.set_title("Cluster CPU Utilisation Over Time — GRU vs HPA")
    ax.set_xlim(0, max(t_g.max(), t_h.max()) + 5)
    ax.legend(loc="upper right")
    _annotate_bursts(ax)
    fig.tight_layout()
    out = RESULTS_DIR / "cpu_timeline.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_scaling_lag(stats):
    labels = ["Scale-up lag\n(Spike 1)", "Scale-up lag\n(Spike 2)",
              "Scale-down lag\n(Drop 1)", "Scale-down lag\n(Drop 2)"]
    keys_g = ["gru_scaleup_lag_s1", "gru_scaleup_lag_s2",
               "gru_scaledown_lag_d1", "gru_scaledown_lag_d2"]
    keys_h = ["hpa_scaleup_lag_s1", "hpa_scaleup_lag_s2",
               "hpa_scaledown_lag_d1", "hpa_scaledown_lag_d2"]

    gru_vals = [_safe_val(stats, k) for k in keys_g]
    hpa_vals = [_safe_val(stats, k) for k in keys_h]
    gru_err  = [_safe_err(stats, k) for k in keys_g]
    hpa_err  = [_safe_err(stats, k) for k in keys_h]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_g = ax.bar(x - w/2, gru_vals, w, color=GRU_COLOR, label="GRU autoscaler",
                    yerr=gru_err, error_kw=_error_kw(), zorder=3)
    bars_h = ax.bar(x + w/2, hpa_vals, w, color=HPA_COLOR, label="HPA (reactive)",
                    alpha=0.85, yerr=hpa_err, error_kw=_error_kw(), zorder=3)

    for bars in (bars_g, bars_h):
        for bar in bars:
            h_ = bar.get_height()
            if np.isnan(h_):
                ax.text(bar.get_x() + bar.get_width()/2, 2, "N/A",
                        ha="center", va="bottom", fontsize=9, color="#555")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, h_ + 1,
                        f"{h_:.0f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Lag (seconds)")
    ax.set_title("Autoscaler Response Lag — GRU vs HPA\n"
                 "(lower = faster reaction = better)")
    _add_run_note(ax, stats)
    ax.legend(); ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    out = RESULTS_DIR / "scaling_lag.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_over_provisioning(stats):
    labels = ["Recovery\n(after Drop 1)", "Cooldown\n(after Drop 2)", "Total"]
    keys_g = ["gru_excess_rs_recovery", "gru_excess_rs_cooldown", "gru_excess_rs_total"]
    keys_h = ["hpa_excess_rs_recovery", "hpa_excess_rs_cooldown", "hpa_excess_rs_total"]

    gru_vals = [_safe_val(stats, k) for k in keys_g]
    hpa_vals = [_safe_val(stats, k) for k in keys_h]
    gru_err  = [_safe_err(stats, k) for k in keys_g]
    hpa_err  = [_safe_err(stats, k) for k in keys_h]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, gru_vals, w, color=GRU_COLOR, label="GRU autoscaler",
           yerr=gru_err, error_kw=_error_kw(), zorder=3)
    ax.bar(x + w/2, hpa_vals, w, color=HPA_COLOR, label="HPA (reactive)",
           alpha=0.85, yerr=hpa_err, error_kw=_error_kw(), zorder=3)

    ymax = max(v for v in gru_vals + hpa_vals if not np.isnan(v)) if any(
        not np.isnan(v) for v in gru_vals + hpa_vals) else 1
    for xi, (g, h_) in enumerate(zip(gru_vals, hpa_vals)):
        for xpos, val in [(xi - w/2, g), (xi + w/2, h_)]:
            if not np.isnan(val):
                ax.text(xpos, val + ymax * 0.015, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Excess replica-seconds (replica·s above minimum)")
    ax.set_title("Over-provisioning After Load Drop — GRU vs HPA\n"
                 "(lower = less wasted resource = better)")
    _add_run_note(ax, stats)
    ax.legend(); ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    out = RESULTS_DIR / "over_provisioning.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_under_provisioning(stats):
    """Under-provisioning time = seconds where CPU/pod > threshold.

    This is the primary SLA-violation proxy.  HPA's scale-up lag means pods
    run at or above their CPU throttling point for tens of seconds per spike.
    GRU's predictive scale-up reduces this window significantly.
    """
    thr = CPU_UNDERPROVISIONED_THRESHOLD_M
    labels = [f"Spike 1\n({SPIKE1_PEAK}–{SPIKE1_END}s)",
              f"Spike 2\n({SPIKE2_PEAK}–{SPIKE2_END}s)",
              "Total"]
    keys_g = ["gru_underprov_s1", "gru_underprov_s2", "gru_underprov_total"]
    keys_h = ["hpa_underprov_s1", "hpa_underprov_s2", "hpa_underprov_total"]

    gru_vals = [_safe_val(stats, k) for k in keys_g]
    hpa_vals = [_safe_val(stats, k) for k in keys_h]
    gru_err  = [_safe_err(stats, k) for k in keys_g]
    hpa_err  = [_safe_err(stats, k) for k in keys_h]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, gru_vals, w, color=GRU_COLOR, label="GRU autoscaler",
           yerr=gru_err, error_kw=_error_kw(), zorder=3)
    ax.bar(x + w/2, hpa_vals, w, color=HPA_COLOR, label="HPA (reactive)",
           alpha=0.85, yerr=hpa_err, error_kw=_error_kw(), zorder=3)

    ymax = max((v for v in gru_vals + hpa_vals if not np.isnan(v)), default=1)
    for xi, (g, h_) in enumerate(zip(gru_vals, hpa_vals)):
        for xpos, val in [(xi - w/2, g), (xi + w/2, h_)]:
            if not np.isnan(val):
                ax.text(xpos, val + ymax * 0.015, f"{val:.0f}s",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Under-provisioned time (seconds)")
    ax.set_title(
        f"Under-Provisioning Time (CPU/pod > {thr} m) — GRU vs HPA\n"
        "(lower = fewer potential SLA violations = better)"
    )
    _add_run_note(ax, stats)
    ax.legend(); ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    fig.tight_layout()
    out = RESULTS_DIR / "under_provisioning.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_resource_efficiency(runs_g, runs_h):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    g = _best_single_run(runs_g)
    h = _best_single_run(runs_h)
    t_g, r_g, c_g = (_valid(g["elapsed_s"]), _valid(g["replicas"]),
                      g["cpu_total_m"].copy())
    t_h, r_h, c_h = (_valid(h["elapsed_s"]), _valid(h["replicas"]),
                      h["cpu_total_m"].copy())

    eff_g = np.where(r_g > 0, c_g / r_g, np.nan)
    eff_h = np.where(r_h > 0, c_h / r_h, np.nan)
    ax.plot(t_g, eff_g, color=GRU_COLOR, lw=1.8, label="GRU — CPU/replica (m)")
    ax.plot(t_h, eff_h, color=HPA_COLOR,  lw=1.8, label="HPA — CPU/replica (m)", ls="--")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("CPU per replica (millicores)")
    ax.set_title("Resource Efficiency (CPU per Replica) — GRU vs HPA\n"
                 "(higher & more stable = better utilisation)")
    ax.set_xlim(0, max(t_g.max(), t_h.max()) + 5)
    ax.legend(loc="upper right")
    _annotate_bursts(ax)
    fig.tight_layout()
    out = RESULTS_DIR / "resource_efficiency.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out.name}")


def _add_run_note(ax, stats):
    n = stats.get("n_runs", 1)
    if n > 1:
        ax.annotate(f"N={n} runs, error bars = std dev",
                    xy=(0.99, 0.01), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color="#555",
                    style="italic")


def fig_paper_summary(runs_g, runs_h, stats):
    """4×2 combined figure for paper submission."""
    thr = CPU_UNDERPROVISIONED_THRESHOLD_M
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        "GRU Predictive Autoscaler vs Kubernetes HPA — Burst Test Results",
        fontsize=15, fontweight="bold", y=1.005,
    )

    g = _best_single_run(runs_g)
    h = _best_single_run(runs_h)
    t_g, r_g, c_g = _valid(g["elapsed_s"]), _valid(g["replicas"]), g["cpu_total_m"].copy()
    t_h, r_h, c_h = _valid(h["elapsed_s"]), _valid(h["replicas"]), h["cpu_total_m"].copy()
    n = stats.get("n_runs", 1)
    w = 0.35

    # ── (0,0) Replica timeline ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.step(t_g, r_g, where="post", color=GRU_COLOR, lw=2.0, label="GRU")
    ax.step(t_h, r_h, where="post", color=HPA_COLOR,  lw=2.0, label="HPA", ls="--")
    ax.set_title("(a) Replica Count Timeline")
    ax.set_xlabel("Elapsed (s)"); ax.set_ylabel("Replicas")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=9); _annotate_bursts(ax)

    # ── (0,1) CPU timeline ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_g, c_g, color=GRU_COLOR, lw=1.8, label="GRU")
    ax.plot(t_h, c_h, color=HPA_COLOR,  lw=1.8, label="HPA", ls="--")
    ax.set_title("(b) Cluster CPU Utilisation")
    ax.set_xlabel("Elapsed (s)"); ax.set_ylabel("CPU (millicores)")
    ax.legend(fontsize=9); _annotate_bursts(ax)

    # ── (1,0) Scale-up lag ─────────────────────────────────────────────────
    ax = axes[1, 0]
    gv = [_safe_val(stats, k) for k in ["gru_scaleup_lag_s1", "gru_scaleup_lag_s2"]]
    hv = [_safe_val(stats, k) for k in ["hpa_scaleup_lag_s1", "hpa_scaleup_lag_s2"]]
    ge = [_safe_err(stats, k) for k in ["gru_scaleup_lag_s1", "gru_scaleup_lag_s2"]]
    he = [_safe_err(stats, k) for k in ["hpa_scaleup_lag_s1", "hpa_scaleup_lag_s2"]]
    x2 = np.arange(2)
    ax.bar(x2 - w/2, gv, w, color=GRU_COLOR, label="GRU", yerr=ge, error_kw=_error_kw(), zorder=3)
    ax.bar(x2 + w/2, hv, w, color=HPA_COLOR, label="HPA", alpha=0.85,
           yerr=he, error_kw=_error_kw(), zorder=3)
    for xi, (a_, b_) in enumerate(zip(gv, hv)):
        for xpos, val in [(xi-w/2, a_), (xi+w/2, b_)]:
            if not np.isnan(val):
                ax.text(xpos, val+0.5, f"{val:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x2); ax.set_xticklabels(["Spike 1", "Spike 2"])
    ax.set_ylabel("Lag (s)"); ax.set_title("(c) Scale-Up Response Lag")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=GRID_ALPHA)
    if n > 1:
        _add_run_note(ax, stats)

    # ── (1,1) Scale-down lag ───────────────────────────────────────────────
    ax = axes[1, 1]
    gv = [_safe_val(stats, k) for k in ["gru_scaledown_lag_d1", "gru_scaledown_lag_d2"]]
    hv = [_safe_val(stats, k) for k in ["hpa_scaledown_lag_d1", "hpa_scaledown_lag_d2"]]
    ge = [_safe_err(stats, k) for k in ["gru_scaledown_lag_d1", "gru_scaledown_lag_d2"]]
    he = [_safe_err(stats, k) for k in ["hpa_scaledown_lag_d1", "hpa_scaledown_lag_d2"]]
    ax.bar(x2 - w/2, gv, w, color=GRU_COLOR, label="GRU", yerr=ge, error_kw=_error_kw(), zorder=3)
    ax.bar(x2 + w/2, hv, w, color=HPA_COLOR, label="HPA", alpha=0.85,
           yerr=he, error_kw=_error_kw(), zorder=3)
    for xi, (a_, b_) in enumerate(zip(gv, hv)):
        for xpos, val in [(xi-w/2, a_), (xi+w/2, b_)]:
            if not np.isnan(val):
                ax.text(xpos, val+0.5, f"{val:.0f}s", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(xpos, 2, "N/A", ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_xticks(x2); ax.set_xticklabels(["Drop 1", "Drop 2"])
    ax.set_ylabel("Lag (s)")
    ax.set_title("(d) Scale-Down Response Lag\n(N/A = stabilisation window active)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=GRID_ALPHA)
    if n > 1:
        _add_run_note(ax, stats)

    # ── (2,0) Over-provisioning ────────────────────────────────────────────
    ax = axes[2, 0]
    gv = [_safe_val(stats, k) for k in
          ["gru_excess_rs_recovery", "gru_excess_rs_cooldown", "gru_excess_rs_total"]]
    hv = [_safe_val(stats, k) for k in
          ["hpa_excess_rs_recovery", "hpa_excess_rs_cooldown", "hpa_excess_rs_total"]]
    ge = [_safe_err(stats, k) for k in
          ["gru_excess_rs_recovery", "gru_excess_rs_cooldown", "gru_excess_rs_total"]]
    he = [_safe_err(stats, k) for k in
          ["hpa_excess_rs_recovery", "hpa_excess_rs_cooldown", "hpa_excess_rs_total"]]
    x3 = np.arange(3)
    ax.bar(x3 - w/2, gv, w, color=GRU_COLOR, label="GRU", yerr=ge, error_kw=_error_kw(), zorder=3)
    ax.bar(x3 + w/2, hv, w, color=HPA_COLOR, label="HPA", alpha=0.85,
           yerr=he, error_kw=_error_kw(), zorder=3)
    ymax_op = max((v for v in gv + hv if not np.isnan(v)), default=1)
    for xi, (a_, b_) in enumerate(zip(gv, hv)):
        for xpos, val in [(xi-w/2, a_), (xi+w/2, b_)]:
            if not np.isnan(val):
                ax.text(xpos, val + ymax_op * 0.01, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x3); ax.set_xticklabels(["Recovery", "Cooldown", "Total"])
    ax.set_ylabel("Excess replica·s")
    ax.set_title("(e) Over-provisioning (replica-seconds wasted)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=GRID_ALPHA)
    if n > 1:
        _add_run_note(ax, stats)

    # ── (2,1) Under-provisioning (NEW — key SLA metric) ───────────────────
    ax = axes[2, 1]
    gv = [_safe_val(stats, k) for k in
          ["gru_underprov_s1", "gru_underprov_s2", "gru_underprov_total"]]
    hv = [_safe_val(stats, k) for k in
          ["hpa_underprov_s1", "hpa_underprov_s2", "hpa_underprov_total"]]
    ge = [_safe_err(stats, k) for k in
          ["gru_underprov_s1", "gru_underprov_s2", "gru_underprov_total"]]
    he = [_safe_err(stats, k) for k in
          ["hpa_underprov_s1", "hpa_underprov_s2", "hpa_underprov_total"]]
    ax.bar(x3 - w/2, gv, w, color=GRU_COLOR, label="GRU", yerr=ge, error_kw=_error_kw(), zorder=3)
    ax.bar(x3 + w/2, hv, w, color=HPA_COLOR, label="HPA", alpha=0.85,
           yerr=he, error_kw=_error_kw(), zorder=3)
    ymax_up = max((v for v in gv + hv if not np.isnan(v)), default=1)
    for xi, (a_, b_) in enumerate(zip(gv, hv)):
        for xpos, val in [(xi-w/2, a_), (xi+w/2, b_)]:
            if not np.isnan(val):
                ax.text(xpos, val + ymax_up * 0.015, f"{val:.0f}s",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x3); ax.set_xticklabels(["Spike 1", "Spike 2", "Total"])
    ax.set_ylabel("Under-provisioned time (s)")
    ax.set_title(f"(f) Under-Provisioning Time\n"
                 f"(CPU/pod > {thr} m — SLA violation proxy)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=GRID_ALPHA)
    if n > 1:
        _add_run_note(ax, stats)

    # ── (3,0) Resource efficiency timeline ────────────────────────────────
    ax = axes[3, 0]
    eff_g = np.where(r_g > 0, c_g / r_g, np.nan)
    eff_h = np.where(r_h > 0, c_h / r_h, np.nan)
    ax.plot(t_g, eff_g, color=GRU_COLOR, lw=1.8, label="GRU")
    ax.plot(t_h, eff_h, color=HPA_COLOR,  lw=1.8, label="HPA", ls="--")
    ax.set_title("(g) CPU per Replica (Resource Efficiency)")
    ax.set_xlabel("Elapsed (s)"); ax.set_ylabel("CPU/replica (m)")
    ax.legend(fontsize=9); _annotate_bursts(ax)

    # ── (3,1) Key statistics text table ───────────────────────────────────
    ax = axes[3, 1]
    ax.axis("off")

    def _fv(key, unit="s", p=1):
        v   = _safe_val(stats, key)
        std = _safe_err(stats, key)
        if np.isnan(v):
            return "N/A"
        s = f"{v:.{p}f} {unit}".strip()
        if n > 1 and std > 0:
            s += f" ±{std:.{p}f}"
        return s

    thr = CPU_UNDERPROVISIONED_THRESHOLD_M
    table_lines = [
        ("Metric", "GRU", "HPA"),
        ("─" * 14, "─" * 8, "─" * 8),
        ("Scale-up lag S1",
         _fv("gru_scaleup_lag_s1"),   _fv("hpa_scaleup_lag_s1")),
        ("Scale-up lag S2",
         _fv("gru_scaleup_lag_s2"),   _fv("hpa_scaleup_lag_s2")),
        ("Scale-down lag D1",
         _fv("gru_scaledown_lag_d1"), _fv("hpa_scaledown_lag_d1")),
        ("─" * 14, "─" * 8, "─" * 8),
        ("Over-prov (r·s)",
         _fv("gru_excess_rs_total", "r·s", 0),
         _fv("hpa_excess_rs_total", "r·s", 0)),
        ("─" * 14, "─" * 8, "─" * 8),
        (f"Under-prov S1",
         _fv("gru_underprov_s1"), _fv("hpa_underprov_s1")),
        (f"Under-prov S2",
         _fv("gru_underprov_s2"), _fv("hpa_underprov_s2")),
        (f"Under-prov total",
         _fv("gru_underprov_total"), _fv("hpa_underprov_total")),
    ]
    col_x  = [0.0, 0.52, 0.78]
    row_h  = 0.082
    y_start = 0.97
    for ri, row in enumerate(table_lines):
        y = y_start - ri * row_h
        for ci, cell in enumerate(row):
            weight = "bold" if ri == 0 else "normal"
            ax.text(col_x[ci], y, cell, transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="left", fontweight=weight,
                    fontfamily="monospace")

    run_info = f"N = {n} run{'s' if n > 1 else ''}  |  ±std shown in table above"
    ax.text(0.0, y_start - len(table_lines) * row_h - 0.04,
            run_info, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#555", style="italic")
    ax.set_title("(h) Key Statistics Summary", pad=6)

    # shared legend
    spike_patch = mpatches.Patch(color=SPIKE_COLOR, alpha=0.6, label="300-VU spike")
    fig.legend(handles=[spike_patch], loc="lower center", ncol=1,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout()
    out = RESULTS_DIR / "paper_summary_figure.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out.name}")


# ── summary text ──────────────────────────────────────────────────────────────

def _fmt(v, unit="s", precision=1):
    if _nanval(v):
        return "N/A"
    return f"{v:.{precision}f} {unit}".strip()


def _fmt_std(stats, key, unit="s", precision=1):
    v   = _safe_val(stats, key)
    std = _safe_err(stats, key)
    n   = stats.get("n_runs", 1)
    if _nanval(v):
        return "N/A"
    s = f"{v:.{precision}f}"
    if n > 1 and std > 0:
        s += f" ±{std:.{precision}f}"
    return f"{s} {unit}".strip()


def write_summary(stats):
    n   = stats.get("n_runs", 1)
    thr = CPU_UNDERPROVISIONED_THRESHOLD_M
    lines = []
    a = lines.append

    a("=" * 68)
    a("  Experiment Results Summary — GRU Autoscaler vs Kubernetes HPA")
    a("  Load pattern: k6 burst test  (kubernetes/k6-burst-test.js)")
    a(f"  Statistical basis: {n} run{'s' if n > 1 else ''}"
      + (" — values shown as mean ± std" if n > 1 else ""))
    a("=" * 68)
    a("")
    a("  Workload stages:")
    a("    t=  0-30 s  baseline        20 VUs")
    a("    t= 30-35 s  spike-1 ramp   300 VUs")
    a("    t= 35-95 s  spike-1 peak   300 VUs  ← key window")
    a("    t= 95-100s  drop-1           5 VUs")
    a("    t=100-160s  recovery        20 VUs  ← over-provisioning window")
    a("    t=160-165s  spike-2 ramp   300 VUs")
    a("    t=165-225s  spike-2 peak   300 VUs")
    a("    t=225-230s  drop-2           5 VUs")
    a("    t=230-260s  cooldown        20 VUs")
    a("")

    col_w = 40
    a(f"  {'Metric':<{col_w}}  {'GRU':>14}  {'HPA':>14}  {'Better':>8}")
    a("  " + "-" * (col_w + 42))

    def row(label, gkey, hkey, unit="s", lower_better=True, precision=1):
        gv = _safe_val(stats, gkey)
        hv = _safe_val(stats, hkey)
        gf = _fmt_std(stats, gkey, unit, precision)
        hf = _fmt_std(stats, hkey, unit, precision)
        if _nanval(gv) or _nanval(hv):
            winner = "?"
        else:
            winner = "GRU ✓" if (gv < hv) == lower_better else "HPA"
        a(f"  {label:<{col_w}}  {gf:>14}  {hf:>14}  {winner:>8}")

    a("  BASELINE")
    a(f"  {'Replicas at baseline':<{col_w}}  "
      f"{_fmt(stats.get('gru_baseline_replicas', np.nan), '', 0):>14}  "
      f"{_fmt(stats.get('hpa_baseline_replicas', np.nan), '', 0):>14}")
    a("")

    a("  SCALE-UP LAG  (seconds from spike start → first new replica)")
    a("  [GRU predicts load in advance; HPA waits for CPU to rise]")
    row("  Spike 1",
        "gru_scaleup_lag_s1", "hpa_scaleup_lag_s1")
    row("  Spike 2",
        "gru_scaleup_lag_s2", "hpa_scaleup_lag_s2")
    a("")

    a("  PEAK REPLICAS REACHED DURING SPIKE")
    row("  Spike 1 peak replicas",
        "gru_peak_replicas_s1", "hpa_peak_replicas_s1", unit="", lower_better=False, precision=0)
    row("  Spike 2 peak replicas",
        "gru_peak_replicas_s2", "hpa_peak_replicas_s2", unit="", lower_better=False, precision=0)
    a("")

    a("  SCALE-DOWN LAG  (seconds from drop start → first replica released)")
    a("  [HPA stabilizationWindowSeconds=300 holds replicas for up to 5 min]")
    row("  Drop 1",
        "gru_scaledown_lag_d1", "hpa_scaledown_lag_d1")
    row("  Drop 2",
        "gru_scaledown_lag_d2", "hpa_scaledown_lag_d2")
    a("")

    a("  OVER-PROVISIONING  (excess replica-seconds above minimum)")
    a("  [wasted compute cost during recovery/cooldown windows]")
    row("  Recovery window (t=95–160 s)",
        "gru_excess_rs_recovery", "hpa_excess_rs_recovery", unit="r·s", precision=0)
    row("  Cooldown window (t=225–260 s)",
        "gru_excess_rs_cooldown", "hpa_excess_rs_cooldown", unit="r·s", precision=0)
    row("  TOTAL excess replica-seconds",
        "gru_excess_rs_total",    "hpa_excess_rs_total",    unit="r·s", precision=0)
    a("")

    a(f"  UNDER-PROVISIONING  (seconds where CPU/pod > {thr} m)")
    a(f"  [CPU throttling risk / SLA-violation proxy — STRONGEST ARGUMENT]")
    a(f"  [Pod CPU limit = 1000 m; threshold = {thr} m = {thr/10:.0f}% of limit]")
    row("  Spike 1",
        "gru_underprov_s1", "hpa_underprov_s1")
    row("  Spike 2",
        "gru_underprov_s2", "hpa_underprov_s2")
    row("  TOTAL under-provisioned seconds",
        "gru_underprov_total", "hpa_underprov_total")
    a("")

    a("  PEAK CPU PER POD DURING SPIKE 1")
    row("  Peak CPU / pod",
        "gru_peak_cpu_per_pod_s1", "hpa_peak_cpu_per_pod_s1", unit="m", precision=0)
    a("")
    a("=" * 68)

    text = "\n".join(lines)
    out  = RESULTS_DIR / "experiment_summary.txt"
    out.write_text(text)
    print(f"  Saved: {out.name}")
    print()
    print(text)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  plot_experiment_results.py")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    print("  Loading GRU run data…")
    runs_g, n_g = load_scenario_runs("gru")
    print("  Loading HPA run data…")
    runs_h, n_h = load_scenario_runs("hpa")

    if n_g != n_h:
        print(f"\nWARNING: GRU has {n_g} run(s) but HPA has {n_h} run(s).")
        n = min(n_g, n_h)
        print(f"  Using the first {n} run(s) of each for matched analysis.")
        runs_g, runs_h = runs_g[:n], runs_h[:n]

    print()
    print("  Computing statistics…")
    stats = compute_stats_multi(runs_g, runs_h)

    print("  Generating figures…")
    fig_replica_timeline(runs_g, runs_h)
    fig_cpu_timeline(runs_g, runs_h)
    fig_scaling_lag(stats)
    fig_over_provisioning(stats)
    fig_under_provisioning(stats)
    fig_resource_efficiency(runs_g, runs_h)
    fig_paper_summary(runs_g, runs_h, stats)
    write_summary(stats)

    n_runs = stats["n_runs"]
    print()
    print("✅  All outputs written to results/")
    print(f"    Statistical basis: {n_runs} run{'s' if n_runs > 1 else ''}"
          + (" (mean ± std)" if n_runs > 1 else ""))
    print("    Figures for paper:")
    for name in [
        "replica_timeline.png",
        "cpu_timeline.png",
        "scaling_lag.png",
        "over_provisioning.png",
        "under_provisioning.png",
        "resource_efficiency.png",
        "paper_summary_figure.png",
        "experiment_summary.txt",
    ]:
        print(f"      results/{name}")


if __name__ == "__main__":
    main()
