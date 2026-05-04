#!/usr/bin/env python3
"""
plot_experiment_results.py
--------------------------
Read the two experiment CSVs produced by collect_experiment.py and generate
publication-quality figures plus a summary statistics file that can be copied
directly into a research paper.

Usage
-----
  python plot_experiment_results.py

Prerequisites
-------------
  results/gru_experiment.csv   (from: python collect_experiment.py --scenario gru)
  results/hpa_experiment.csv   (from: python collect_experiment.py --scenario hpa)

Outputs written to results/
---------------------------
  replica_timeline.png       — Figure 1: replica count over time, GRU vs HPA
  cpu_timeline.png           — Figure 2: cluster CPU usage over time, GRU vs HPA
  scaling_lag.png            — Figure 3: scale-up and scale-down lag bar chart
  over_provisioning.png      — Figure 4: excess replica-seconds during recovery
  resource_efficiency.png    — Figure 5: CPU-per-replica (utilisation density)
  paper_summary_figure.png   — Figure 6: 3×2 combined figure for the paper
  experiment_summary.txt     — Numeric results table for the paper

Burst event timeline (from kubernetes/k6-burst-test.js)
--------------------------------------------------------
  t=  0s  baseline        20 VUs
  t= 30s  spike-1 ramp   300 VUs
  t= 35s  spike-1 peak   300 VUs
  t= 95s  drop-1           5 VUs
  t=100s  recovery         20 VUs
  t=160s  spike-2 ramp   300 VUs
  t=165s  spike-2 peak   300 VUs
  t=225s  drop-2           5 VUs
  t=230s  cooldown         20 VUs
  t=260s  end
"""

import os
import sys
import csv
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"

GRU_CSV = RESULTS_DIR / "gru_experiment.csv"
HPA_CSV = RESULTS_DIR / "hpa_experiment.csv"

# ── burst event timestamps (seconds from experiment start) ────────────────────

SPIKE1_START   =  30   # k6 ramps to 300 VUs
SPIKE1_PEAK    =  35   # 300 VUs sustained
SPIKE1_END     =  95   # k6 drops to 5 VUs
RECOVERY_END   = 160   # recovery window ends
SPIKE2_START   = 160
SPIKE2_PEAK    = 165
SPIKE2_END     = 225
COOLDOWN_START = 230
TEST_END       = 260

SPIKE_REGIONS  = [(SPIKE1_PEAK, SPIKE1_END), (SPIKE2_PEAK, SPIKE2_END)]
DROP_REGIONS   = [(SPIKE1_END, RECOVERY_END), (SPIKE2_END, TEST_END)]

# ── plot style ────────────────────────────────────────────────────────────────

GRU_COLOR  = "#1f77b4"   # blue
HPA_COLOR  = "#d62728"   # red
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
    """Return a dict of lists keyed by column name.

    Missing / empty cells are stored as np.nan.
    """
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("  Run: python collect_experiment.py --scenario gru")
        print("  Run: python collect_experiment.py --scenario hpa")
        sys.exit(1)

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


# ── analysis helpers ──────────────────────────────────────────────────────────

def _valid(arr):
    """Return array with nan replaced by forward-fill then 0."""
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last = out[i]
        elif not np.isnan(last):
            out[i] = last
    out = np.where(np.isnan(out), 0, out)
    return out


def find_first_scale_up(t, replicas, spike_start, baseline_replicas):
    """Return seconds-after-spike-start when replicas first exceed baseline."""
    mask = t >= spike_start
    for i in np.where(mask)[0]:
        if replicas[i] > baseline_replicas:
            return t[i] - spike_start
    return np.nan


def find_first_scale_down(t, replicas, drop_start, peak_replicas):
    """Return seconds-after-drop-start when replicas first drop below peak."""
    mask = t >= drop_start
    for i in np.where(mask)[0]:
        if replicas[i] < peak_replicas:
            return t[i] - drop_start
    return np.nan


def excess_replica_seconds(t, replicas, window_start, window_end, min_replicas):
    """Integrate (replicas - min_replicas) over a time window (trapezoidal)."""
    mask = (t >= window_start) & (t <= window_end)
    tw = t[mask]
    rw = replicas[mask]
    if len(tw) < 2:
        return 0.0
    excess = np.maximum(rw - min_replicas, 0)
    return float(np.trapezoid(excess, tw))


def peak_cpu_during_spike(t, cpu, spike_start, spike_end):
    mask = (t >= spike_start) & (t <= spike_end)
    vals = cpu[mask]
    vals = vals[~np.isnan(vals)]
    return float(np.nanmax(vals)) if len(vals) > 0 else np.nan


# ── shared annotation helper ──────────────────────────────────────────────────

def annotate_bursts(ax, ymax_frac=1.0, label_alpha=0.6):
    """Draw shaded spike regions and vertical drop lines."""
    ylim = ax.get_ylim()
    yspan = ylim[1] - ylim[0]

    for (s, e) in SPIKE_REGIONS:
        ax.axvspan(s, e, color=SPIKE_COLOR, alpha=0.45, zorder=0, label="_spike")

    for t_drop in [SPIKE1_END, SPIKE2_END]:
        ax.axvline(t_drop, color="#888888", ls="--", lw=1.0, alpha=0.7)

    for t_ev, label in [
        (SPIKE1_START, "Spike 1"), (SPIKE1_END, "Drop 1"),
        (SPIKE2_START, "Spike 2"), (SPIKE2_END, "Drop 2"),
    ]:
        ax.text(
            t_ev + 2, ylim[0] + yspan * 0.96,
            label, fontsize=8, color=ANNOT_COLOR, alpha=label_alpha,
            va="top", ha="left",
        )


# ── individual figures ────────────────────────────────────────────────────────

def fig_replica_timeline(gru, hpa):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    t_g = _valid(gru["elapsed_s"])
    r_g = _valid(gru["replicas"])
    t_h = _valid(hpa["elapsed_s"])
    r_h = _valid(hpa["replicas"])

    ax.step(t_g, r_g, where="post", color=GRU_COLOR, lw=2.0,
            label="GRU autoscaler", zorder=3)
    ax.step(t_h, r_h, where="post", color=HPA_COLOR,  lw=2.0,
            label="HPA (reactive)", zorder=3, ls="--")

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Replica count")
    ax.set_title("Replica Count Over Time — GRU vs HPA (k6 burst test)")
    ax.set_xlim(0, max(t_g.max(), t_h.max()) + 5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="upper right")
    annotate_bursts(ax)

    fig.tight_layout()
    out = RESULTS_DIR / "replica_timeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_cpu_timeline(gru, hpa):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    t_g = _valid(gru["elapsed_s"])
    c_g = gru["cpu_total_m"].copy()
    t_h = _valid(hpa["elapsed_s"])
    c_h = hpa["cpu_total_m"].copy()

    ax.plot(t_g, c_g, color=GRU_COLOR, lw=1.8, label="GRU — cluster CPU (m)")
    ax.plot(t_h, c_h, color=HPA_COLOR,  lw=1.8, label="HPA — cluster CPU (m)",
            ls="--")

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Total cluster CPU (millicores)")
    ax.set_title("Cluster CPU Utilisation Over Time — GRU vs HPA")
    ax.set_xlim(0, max(t_g.max(), t_h.max()) + 5)
    ax.legend(loc="upper right")
    annotate_bursts(ax)

    fig.tight_layout()
    out = RESULTS_DIR / "cpu_timeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_scaling_lag(stats):
    labels = ["Scale-up lag\n(Spike 1)", "Scale-up lag\n(Spike 2)",
              "Scale-down lag\n(Drop 1)", "Scale-down lag\n(Drop 2)"]
    gru_vals = [
        stats["gru_scaleup_lag_s1"], stats["gru_scaleup_lag_s2"],
        stats["gru_scaledown_lag_d1"], stats["gru_scaledown_lag_d2"],
    ]
    hpa_vals = [
        stats["hpa_scaleup_lag_s1"], stats["hpa_scaleup_lag_s2"],
        stats["hpa_scaledown_lag_d1"], stats["hpa_scaledown_lag_d2"],
    ]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_g = ax.bar(x - w/2, gru_vals, w, color=GRU_COLOR, label="GRU autoscaler",
                    zorder=3)
    bars_h = ax.bar(x + w/2, hpa_vals, w, color=HPA_COLOR,  label="HPA (reactive)",
                    zorder=3, alpha=0.85)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, 2, "N/A",
                        ha="center", va="bottom", fontsize=9, color="#555")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f"{h:.0f}s", ha="center", va="bottom", fontsize=9)

    _label_bars(bars_g)
    _label_bars(bars_h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Lag (seconds)")
    ax.set_title("Autoscaler Response Lag — GRU vs HPA\n"
                 "(lower = faster reaction = better)")
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    fig.tight_layout()
    out = RESULTS_DIR / "scaling_lag.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_over_provisioning(stats):
    labels = ["Recovery\n(after Drop 1)", "Cooldown\n(after Drop 2)", "Total"]
    gru_vals = [
        stats["gru_excess_rs_recovery"],
        stats["gru_excess_rs_cooldown"],
        stats["gru_excess_rs_total"],
    ]
    hpa_vals = [
        stats["hpa_excess_rs_recovery"],
        stats["hpa_excess_rs_cooldown"],
        stats["hpa_excess_rs_total"],
    ]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, gru_vals, w, color=GRU_COLOR, label="GRU autoscaler", zorder=3)
    ax.bar(x + w/2, hpa_vals, w, color=HPA_COLOR,  label="HPA (reactive)",
           zorder=3, alpha=0.85)

    for xi, (g, h) in enumerate(zip(gru_vals, hpa_vals)):
        for xpos, val in [(xi - w/2, g), (xi + w/2, h)]:
            ax.text(xpos, val + max(gru_vals + hpa_vals) * 0.015,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Excess replica-seconds\n(replica·s above minimum)")
    ax.set_title("Over-provisioning After Load Drop — GRU vs HPA\n"
                 "(lower = less wasted resource = better)")
    ax.legend()
    ax.grid(axis="y", alpha=GRID_ALPHA)

    fig.tight_layout()
    out = RESULTS_DIR / "over_provisioning.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_resource_efficiency(gru, hpa):
    """CPU-per-replica: a well-scaled system should keep this high and stable."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    t_g = _valid(gru["elapsed_s"])
    t_h = _valid(hpa["elapsed_s"])

    r_g = _valid(gru["replicas"])
    r_h = _valid(hpa["replicas"])
    c_g = gru["cpu_total_m"].copy()
    c_h = hpa["cpu_total_m"].copy()

    # Avoid division by zero
    eff_g = np.where(r_g > 0, c_g / r_g, np.nan)
    eff_h = np.where(r_h > 0, c_h / r_h, np.nan)

    ax.plot(t_g, eff_g, color=GRU_COLOR, lw=1.8, label="GRU — CPU/replica (m)")
    ax.plot(t_h, eff_h, color=HPA_COLOR,  lw=1.8, label="HPA — CPU/replica (m)",
            ls="--")

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("CPU per replica (millicores)")
    ax.set_title("Resource Efficiency (CPU per Replica) — GRU vs HPA\n"
                 "(higher & more stable = better utilisation)")
    ax.set_xlim(0, max(t_g.max(), t_h.max()) + 5)
    ax.legend(loc="upper right")
    annotate_bursts(ax)

    fig.tight_layout()
    out = RESULTS_DIR / "resource_efficiency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig_paper_summary(gru, hpa, stats):
    """3×2 combined figure suitable for a single paper figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        "GRU Predictive Autoscaler vs Kubernetes HPA — Burst Test Results",
        fontsize=15, fontweight="bold", y=1.01,
    )

    t_g = _valid(gru["elapsed_s"])
    r_g = _valid(gru["replicas"])
    t_h = _valid(hpa["elapsed_s"])
    r_h = _valid(hpa["replicas"])
    c_g = gru["cpu_total_m"].copy()
    c_h = hpa["cpu_total_m"].copy()

    # ── (0,0) Replica timeline ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.step(t_g, r_g, where="post", color=GRU_COLOR, lw=2.0, label="GRU")
    ax.step(t_h, r_h, where="post", color=HPA_COLOR,  lw=2.0, label="HPA", ls="--")
    ax.set_title("(a) Replica Count Timeline")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Replicas")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=9)
    annotate_bursts(ax)

    # ── (0,1) CPU timeline ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_g, c_g, color=GRU_COLOR, lw=1.8, label="GRU")
    ax.plot(t_h, c_h, color=HPA_COLOR,  lw=1.8, label="HPA", ls="--")
    ax.set_title("(b) Cluster CPU Utilisation")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("CPU (millicores)")
    ax.legend(fontsize=9)
    annotate_bursts(ax)

    # ── (1,0) Scale-up lag ─────────────────────────────────────────────────
    ax = axes[1, 0]
    labels_up = ["Spike 1", "Spike 2"]
    gru_up = [stats["gru_scaleup_lag_s1"], stats["gru_scaleup_lag_s2"]]
    hpa_up = [stats["hpa_scaleup_lag_s1"], stats["hpa_scaleup_lag_s2"]]
    x = np.arange(len(labels_up))
    w = 0.35
    ax.bar(x - w/2, gru_up, w, color=GRU_COLOR, label="GRU", zorder=3)
    ax.bar(x + w/2, hpa_up, w, color=HPA_COLOR,  label="HPA", zorder=3, alpha=0.85)
    for xi, (g, h) in enumerate(zip(gru_up, hpa_up)):
        for xpos, val in [(xi - w/2, g), (xi + w/2, h)]:
            if not np.isnan(val):
                ax.text(xpos, val + 0.5, f"{val:.0f}s",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels_up)
    ax.set_ylabel("Lag (s)")
    ax.set_title("(c) Scale-Up Response Lag")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # ── (1,1) Scale-down lag ───────────────────────────────────────────────
    ax = axes[1, 1]
    labels_dn = ["Drop 1", "Drop 2"]
    gru_dn = [stats["gru_scaledown_lag_d1"], stats["gru_scaledown_lag_d2"]]
    hpa_dn = [stats["hpa_scaledown_lag_d1"], stats["hpa_scaledown_lag_d2"]]
    ax.bar(x - w/2, gru_dn, w, color=GRU_COLOR, label="GRU", zorder=3)
    ax.bar(x + w/2, hpa_dn, w, color=HPA_COLOR,  label="HPA", zorder=3, alpha=0.85)
    for xi, (g, h) in enumerate(zip(gru_dn, hpa_dn)):
        for xpos, val in [(xi - w/2, g), (xi + w/2, h)]:
            if not np.isnan(val):
                ax.text(xpos, val + 0.5, f"{val:.0f}s",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels_dn)
    ax.set_ylabel("Lag (s)")
    ax.set_title("(d) Scale-Down Response Lag\n(N/A = stabilisation window still active)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # ── (2,0) Over-provisioning ────────────────────────────────────────────
    ax = axes[2, 0]
    labels_op = ["Recovery", "Cooldown", "Total"]
    gru_op = [stats["gru_excess_rs_recovery"],
               stats["gru_excess_rs_cooldown"],
               stats["gru_excess_rs_total"]]
    hpa_op = [stats["hpa_excess_rs_recovery"],
               stats["hpa_excess_rs_cooldown"],
               stats["hpa_excess_rs_total"]]
    x3 = np.arange(len(labels_op))
    ax.bar(x3 - w/2, gru_op, w, color=GRU_COLOR, label="GRU", zorder=3)
    ax.bar(x3 + w/2, hpa_op, w, color=HPA_COLOR,  label="HPA", zorder=3, alpha=0.85)
    for xi, (g, h) in enumerate(zip(gru_op, hpa_op)):
        for xpos, val in [(xi - w/2, g), (xi + w/2, h)]:
            ax.text(xpos, val + max(gru_op + hpa_op) * 0.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x3); ax.set_xticklabels(labels_op)
    ax.set_ylabel("Excess replica·s")
    ax.set_title("(e) Over-provisioning (replica-seconds wasted)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # ── (2,1) CPU-per-replica efficiency ──────────────────────────────────
    ax = axes[2, 1]
    r_g2 = np.where(r_g > 0, r_g, np.nan)
    r_h2 = np.where(r_h > 0, r_h, np.nan)
    eff_g = c_g / r_g2
    eff_h = c_h / r_h2
    ax.plot(t_g, eff_g, color=GRU_COLOR, lw=1.8, label="GRU")
    ax.plot(t_h, eff_h, color=HPA_COLOR,  lw=1.8, label="HPA", ls="--")
    ax.set_title("(f) CPU per Replica (Resource Efficiency)")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("CPU/replica (m)")
    ax.legend(fontsize=9)
    annotate_bursts(ax)

    # shared legend patch for spike regions
    spike_patch = mpatches.Patch(color=SPIKE_COLOR, alpha=0.6, label="300-VU spike")
    fig.legend(handles=[spike_patch], loc="lower center",
               ncol=1, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.015))

    fig.tight_layout()
    out = RESULTS_DIR / "paper_summary_figure.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── summary statistics ────────────────────────────────────────────────────────

def compute_stats(gru, hpa):
    t_g = _valid(gru["elapsed_s"])
    r_g = _valid(gru["replicas"])
    t_h = _valid(hpa["elapsed_s"])
    r_h = _valid(hpa["replicas"])

    # Baseline replica count = mode of first 30 s
    def baseline_r(t, r, window=30):
        mask = t <= window
        vals = r[mask][r[mask] > 0]
        return float(np.median(vals)) if len(vals) else 1.0

    br_g = baseline_r(t_g, r_g)
    br_h = baseline_r(t_h, r_h)

    # Peak replica during each spike
    def peak_r(t, r, s, e):
        mask = (t >= s) & (t <= e)
        return float(np.max(r[mask])) if mask.any() else np.nan

    gru_peak_s1 = peak_r(t_g, r_g, SPIKE1_START, SPIKE1_END)
    hpa_peak_s1 = peak_r(t_h, r_h, SPIKE1_START, SPIKE1_END)
    gru_peak_s2 = peak_r(t_g, r_g, SPIKE2_START, SPIKE2_END)
    hpa_peak_s2 = peak_r(t_h, r_h, SPIKE2_START, SPIKE2_END)

    # Scale-up lag
    gru_su_s1 = find_first_scale_up(t_g, r_g, SPIKE1_START, br_g)
    hpa_su_s1 = find_first_scale_up(t_h, r_h, SPIKE1_START, br_h)
    gru_su_s2 = find_first_scale_up(t_g, r_g, SPIKE2_START, br_g)
    hpa_su_s2 = find_first_scale_up(t_h, r_h, SPIKE2_START, br_h)

    # Scale-down lag
    gru_sd_d1 = find_first_scale_down(t_g, r_g, SPIKE1_END, gru_peak_s1)
    hpa_sd_d1 = find_first_scale_down(t_h, r_h, SPIKE1_END, hpa_peak_s1)
    gru_sd_d2 = find_first_scale_down(t_g, r_g, SPIKE2_END, gru_peak_s2)
    hpa_sd_d2 = find_first_scale_down(t_h, r_h, SPIKE2_END, hpa_peak_s2)

    # Over-provisioning
    min_r = 1.0
    gru_er_rec  = excess_replica_seconds(t_g, r_g, SPIKE1_END,   RECOVERY_END,   min_r)
    hpa_er_rec  = excess_replica_seconds(t_h, r_h, SPIKE1_END,   RECOVERY_END,   min_r)
    gru_er_cool = excess_replica_seconds(t_g, r_g, SPIKE2_END,   TEST_END,        min_r)
    hpa_er_cool = excess_replica_seconds(t_h, r_h, SPIKE2_END,   TEST_END,        min_r)

    # Peak CPU per pod during spikes
    def peak_cpp(t, cpu, pods, s, e):
        mask = (t >= s) & (t <= e)
        cpp  = cpu[mask] / np.where(pods[mask] > 0, pods[mask], np.nan)
        return float(np.nanmax(cpp)) if mask.any() else np.nan

    c_g = gru["cpu_total_m"]
    p_g = _valid(gru["pods_measured"])
    c_h = hpa["cpu_total_m"]
    p_h = _valid(hpa["pods_measured"])

    gru_peak_cpp_s1 = peak_cpp(t_g, c_g, p_g, SPIKE1_START, SPIKE1_END)
    hpa_peak_cpp_s1 = peak_cpp(t_h, c_h, p_h, SPIKE1_START, SPIKE1_END)

    return {
        # baselines
        "gru_baseline_replicas":  br_g,
        "hpa_baseline_replicas":  br_h,
        # peak replicas
        "gru_peak_replicas_s1":   gru_peak_s1,
        "hpa_peak_replicas_s1":   hpa_peak_s1,
        "gru_peak_replicas_s2":   gru_peak_s2,
        "hpa_peak_replicas_s2":   hpa_peak_s2,
        # scale-up lag
        "gru_scaleup_lag_s1":     gru_su_s1,
        "hpa_scaleup_lag_s1":     hpa_su_s1,
        "gru_scaleup_lag_s2":     gru_su_s2,
        "hpa_scaleup_lag_s2":     hpa_su_s2,
        # scale-down lag
        "gru_scaledown_lag_d1":   gru_sd_d1,
        "hpa_scaledown_lag_d1":   hpa_sd_d1,
        "gru_scaledown_lag_d2":   gru_sd_d2,
        "hpa_scaledown_lag_d2":   hpa_sd_d2,
        # over-provisioning
        "gru_excess_rs_recovery": gru_er_rec,
        "hpa_excess_rs_recovery": hpa_er_rec,
        "gru_excess_rs_cooldown": gru_er_cool,
        "hpa_excess_rs_cooldown": hpa_er_cool,
        "gru_excess_rs_total":    gru_er_rec  + gru_er_cool,
        "hpa_excess_rs_total":    hpa_er_rec  + hpa_er_cool,
        # CPU efficiency
        "gru_peak_cpu_per_pod_s1": gru_peak_cpp_s1,
        "hpa_peak_cpu_per_pod_s1": hpa_peak_cpp_s1,
    }


def _fmt(v, unit="s", precision=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{precision}f} {unit}".strip()


def write_summary(stats):
    lines = []
    a = lines.append

    a("=" * 65)
    a("  Experiment Results Summary — GRU Autoscaler vs Kubernetes HPA")
    a("  Load pattern: k6 burst test (kubernetes/k6-burst-test.js)")
    a("=" * 65)
    a("")
    a("  Workload stages (k6 stages):")
    a("    t=  0-30 s  baseline        20 VUs")
    a("    t= 30-35 s  spike-1 ramp   300 VUs")
    a("    t= 35-95 s  spike-1 peak   300 VUs  ← key comparison window")
    a("    t= 95-100s  drop-1           5 VUs")
    a("    t=100-160s  recovery        20 VUs  ← over-provisioning window")
    a("    t=160-165s  spike-2 ramp   300 VUs")
    a("    t=165-225s  spike-2 peak   300 VUs")
    a("    t=225-230s  drop-2           5 VUs")
    a("    t=230-260s  cooldown        20 VUs")
    a("")

    col_w = 36
    a(f"  {'Metric':<{col_w}}  {'GRU':>10}  {'HPA':>10}  {'Better':>8}")
    a("  " + "-" * (col_w + 34))

    def row(label, gv, hv, unit="s", lower_better=True):
        gf = _fmt(gv, unit)
        hf = _fmt(hv, unit)
        if gv is None or hv is None or np.isnan(gv) or np.isnan(hv):
            winner = "?"
        else:
            winner = "GRU ✓" if (gv < hv) == lower_better else "HPA"
        a(f"  {label:<{col_w}}  {gf:>10}  {hf:>10}  {winner:>8}")

    a("  Baseline")
    a(f"  {'Replicas at baseline':<{col_w}}  "
      f"{_fmt(stats['gru_baseline_replicas'], '', 0):>10}  "
      f"{_fmt(stats['hpa_baseline_replicas'], '', 0):>10}")
    a("")

    a("  Scale-up lag (seconds from spike start to first new replica)")
    row("  Spike 1", stats["gru_scaleup_lag_s1"], stats["hpa_scaleup_lag_s1"])
    row("  Spike 2", stats["gru_scaleup_lag_s2"], stats["hpa_scaleup_lag_s2"])
    a("")

    a("  Peak replicas reached during spike")
    row("  Spike 1 peak replicas", stats["gru_peak_replicas_s1"],
        stats["hpa_peak_replicas_s1"], unit="", lower_better=False)
    row("  Spike 2 peak replicas", stats["gru_peak_replicas_s2"],
        stats["hpa_peak_replicas_s2"], unit="", lower_better=False)
    a("")

    a("  Scale-down lag (seconds from drop start to first replica released)")
    a("  (HPA stabilizationWindowSeconds=300 keeps replicas for 5 min)")
    row("  Drop 1", stats["gru_scaledown_lag_d1"], stats["hpa_scaledown_lag_d1"])
    row("  Drop 2", stats["gru_scaledown_lag_d2"], stats["hpa_scaledown_lag_d2"])
    a("")

    a("  Over-provisioning (excess replica-seconds above minimum)")
    row("  Recovery window (t=95-160s)",
        stats["gru_excess_rs_recovery"], stats["hpa_excess_rs_recovery"],
        unit="r·s")
    row("  Cooldown window (t=225-260s)",
        stats["gru_excess_rs_cooldown"], stats["hpa_excess_rs_cooldown"],
        unit="r·s")
    row("  TOTAL excess replica-seconds",
        stats["gru_excess_rs_total"], stats["hpa_excess_rs_total"],
        unit="r·s")
    a("")

    a("  Peak CPU per pod during Spike 1 (lower = better provisioned)")
    row("  Peak CPU / pod (Spike 1)",
        stats["gru_peak_cpu_per_pod_s1"], stats["hpa_peak_cpu_per_pod_s1"],
        unit="m")
    a("")
    a("=" * 65)

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
    print(f"  Reading: {GRU_CSV}")
    print(f"  Reading: {HPA_CSV}")
    print()

    gru = load_csv(GRU_CSV)
    hpa = load_csv(HPA_CSV)

    print("  Computing statistics…")
    stats = compute_stats(gru, hpa)

    print("  Generating figures…")
    RESULTS_DIR.mkdir(exist_ok=True)

    fig_replica_timeline(gru, hpa)
    fig_cpu_timeline(gru, hpa)
    fig_scaling_lag(stats)
    fig_over_provisioning(stats)
    fig_resource_efficiency(gru, hpa)
    fig_paper_summary(gru, hpa, stats)
    write_summary(stats)

    print()
    print("✅  All outputs written to results/")
    print("    Figures for paper:")
    for name in [
        "replica_timeline.png",
        "cpu_timeline.png",
        "scaling_lag.png",
        "over_provisioning.png",
        "resource_efficiency.png",
        "paper_summary_figure.png",
        "experiment_summary.txt",
    ]:
        print(f"      results/{name}")


if __name__ == "__main__":
    main()
