#!/usr/bin/env python3
"""
collect_experiment.py
---------------------
Record replica counts and CPU metrics from the cluster while the k6 burst
test is running.  Run this in a separate terminal for EACH scenario
(GRU autoscaler, then HPA) so that plot_experiment_results.py can compare them.

Usage
-----
# Terminal 1 — start the k6 burst job FIRST, then immediately:
python collect_experiment.py --scenario gru --duration 310

# After the GRU run is finished, swap autoscalers and run again:
python collect_experiment.py --scenario hpa --duration 310

Output files
------------
  results/gru_experiment.csv
  results/hpa_experiment.csv

CSV columns
-----------
  elapsed_s        : seconds since collection started
  timestamp        : ISO-8601 wall-clock time
  replicas         : deployment.spec.replicas (desired)
  ready_replicas   : deployment.status.readyReplicas (running)
  cpu_total_m      : sum of all pod CPU usage in millicores (from metrics-server)
  cpu_per_pod_m    : cpu_total_m / pods_measured
  pods_measured    : number of pods returned by kubectl top

Burst event timeline (from kubernetes/k6-burst-test.js)
--------------------------------------------------------
  t=  0s  baseline       20 VUs
  t= 30s  spike-1 ramp  300 VUs   ← autoscalers should react here
  t= 35s  spike-1 peak  300 VUs
  t= 95s  drop-1          5 VUs   ← scale-down window opens
  t=100s  recovery        20 VUs
  t=160s  spike-2 ramp  300 VUs
  t=165s  spike-2 peak  300 VUs
  t=225s  drop-2          5 VUs
  t=230s  cooldown        20 VUs
  t=260s  end
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

# ── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_NAMESPACE  = "ai-scaler"
DEFAULT_DEPLOYMENT = "cpu-load-app"
DEFAULT_DURATION   = 310   # a little longer than the 260 s k6 test
DEFAULT_INTERVAL   = 5


# ── kubectl helpers ───────────────────────────────────────────────────────────

def _kubectl(*args, timeout=10):
    """Run kubectl and return (stdout, stderr). Returns (None, err) on failure."""
    cmd = ["kubectl"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None, r.stderr.strip()
        return r.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except FileNotFoundError:
        print("ERROR: 'kubectl' not found in PATH.", file=sys.stderr)
        sys.exit(1)


def get_replicas(namespace, deployment):
    """Return (spec_replicas, ready_replicas) or (None, 0)."""
    out, err = _kubectl(
        "get", "deployment", deployment, "-n", namespace,
        "-o", "jsonpath={.spec.replicas},{.status.readyReplicas}",
    )
    if out is None:
        return None, 0
    try:
        parts = out.split(",")
        spec  = int(parts[0]) if parts[0] else None
        ready = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return spec, ready
    except (ValueError, IndexError):
        return None, 0


def get_cpu_top(namespace, label_selector):
    """Return (total_millicores, pod_count) from kubectl top pod.

    Returns (None, 0) if the metrics-server has no data yet.
    """
    out, err = _kubectl(
        "top", "pod", "-n", namespace,
        "-l", label_selector, "--no-headers",
    )
    if out is None or not out:
        return None, 0

    total_m = 0
    count   = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        cpu_str = parts[1]
        try:
            if cpu_str.endswith("m"):
                total_m += int(cpu_str[:-1])
            else:
                # full cores — convert
                total_m += int(float(cpu_str) * 1000)
            count += 1
        except ValueError:
            pass

    return (total_m if count > 0 else None), count


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect cluster metrics during the k6 burst test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario", required=True, choices=["gru", "hpa"],
        help="Which autoscaler is active (gru or hpa)",
    )
    parser.add_argument(
        "--run", type=int, default=None, metavar="N",
        help=(
            "Run number for multi-trial experiments (e.g. --run 1, --run 2, …). "
            "When provided the output file is named "
            "{scenario}_experiment_runN.csv instead of {scenario}_experiment.csv. "
            "Run 3-5 trials to get statistically credible results."
        ),
    )
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION,
        help="How many seconds to collect data",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help="Poll interval in seconds",
    )
    parser.add_argument(
        "--namespace", default=DEFAULT_NAMESPACE,
        help="Kubernetes namespace",
    )
    parser.add_argument(
        "--deployment", default=DEFAULT_DEPLOYMENT,
        help="Deployment name to watch",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.run is not None:
        outfile = os.path.join(RESULTS_DIR, f"{args.scenario}_experiment_run{args.run}.csv")
    else:
        outfile = os.path.join(RESULTS_DIR, f"{args.scenario}_experiment.csv")

    selector = f"app={args.deployment}"

    print("=" * 60)
    run_label = f" (run #{args.run})" if args.run is not None else ""
    print(f"  Experiment Collector — {args.scenario.upper()}{run_label}")
    print("=" * 60)
    print(f"  Scenario   : {args.scenario.upper()}")
    if args.run is not None:
        print(f"  Run        : #{args.run}")
    print(f"  Namespace  : {args.namespace}")
    print(f"  Deployment : {args.deployment}")
    print(f"  Duration   : {args.duration} s")
    print(f"  Interval   : {args.interval} s")
    print(f"  Output     : {outfile}")
    print()
    print("  Start the k6 burst job NOW (if not already running), then")
    print("  this collector will begin in 3 seconds…")
    time.sleep(3)

    start_time = time.time()

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_s", "timestamp",
            "replicas", "ready_replicas",
            "cpu_total_m", "cpu_per_pod_m", "pods_measured",
        ])

        header = (
            f"{'Elapsed':>8}  {'Replicas':>9}  {'Ready':>7}  "
            f"{'CPU_total':>10}  {'CPU/pod':>9}"
        )
        print(header)
        print("-" * len(header))

        while True:
            elapsed = int(time.time() - start_time)
            ts      = datetime.now().isoformat(timespec="seconds")

            replicas, ready   = get_replicas(args.namespace, args.deployment)
            cpu_total, n_pods = get_cpu_top(args.namespace, selector)
            cpu_per_pod = (cpu_total // n_pods) if (cpu_total and n_pods > 0) else None

            writer.writerow([
                elapsed, ts,
                replicas      if replicas  is not None else "",
                ready         if ready     is not None else "",
                cpu_total     if cpu_total is not None else "",
                cpu_per_pod   if cpu_per_pod is not None else "",
                n_pods,
            ])
            f.flush()

            print(
                f"{elapsed:>7}s  "
                f"{str(replicas  or '?'):>9}  "
                f"{str(ready     or '?'):>7}  "
                f"{(str(cpu_total) + 'm') if cpu_total is not None else '?':>10}  "
                f"{(str(cpu_per_pod) + 'm/pod') if cpu_per_pod is not None else '?':>12}"
            )

            if elapsed >= args.duration:
                break

            # Sleep in small chunks so Ctrl-C is responsive
            deadline = time.time() + args.interval
            while time.time() < deadline:
                time.sleep(0.5)

    print()
    print(f"✅  Done. Data saved to: {outfile}")
    print(f"    Run plot_experiment_results.py after BOTH scenarios are complete.")


if __name__ == "__main__":
    main()
