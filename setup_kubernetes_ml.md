# ML Autoscaler — Kubernetes Setup Guide

> **Complete multi-terminal instructions for running, testing, and comparing the ML autoscaler against Kubernetes HPA.**

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Minikube Cluster  (namespace: ai-scaler)                        │
│                                                                  │
│  ┌──────────────┐  HTTP /work   ┌──────────────────────────┐    │
│  │ cpu-load-app │ ◄── loadgen   │  cpu-collector  :8000    │    │
│  │  (workload)  │               │  Metrics API → raw_cpu   │    │
│  └──────────────┘               └────────────┬─────────────┘    │
│                                              │ raw_cpu           │
│                                   ┌──────────▼──────────────┐   │
│                                   │  ml-autoscaler  :5000   │   │
│                                   │  GRU model → replicas   │   │
│                                   └──────────┬──────────────┘   │
└──────────────────────────────────────────────┼──────────────────┘
                                               │ port-forward
                               ┌───────────────▼──────────────┐
                               │  dashboard.py   :8080         │
                               │  http://localhost:8080        │
                               └──────────────────────────────┘
```

| Service | Port | Purpose |
|---------|------|---------|
| `cpu-collector` | 8000 | Reads CPU from Kubernetes Metrics API; maps 0–2 cores → 300–800 |
| `ml-autoscaler` | 5000 | Polls collector, runs GRU prediction, adjusts `cpu-load-app` replicas |
| `dashboard.py` | 8080 | Flask UI polling `/api/metrics`; auto-refreshes every 5 s |
| `cpu-load-app` | 8000 | Target workload hammered by loadgen pods |

---

## Prerequisites (install once)

```bash
# macOS
brew install minikube kubectl

# Python dependencies (run from repo root)
pip install flask requests tensorflow scikit-learn numpy kubernetes

# Verify Docker is running
docker info
```

---

## Terminal Layout

Open **four** terminals side-by-side before you start:

```
┌──────────────────────┬──────────────────────┐
│  Terminal A          │  Terminal B          │
│  Minikube + kubectl  │  Port-forwards       │
│  (setup & monitor)   │  (KEEP RUNNING)      │
├──────────────────────┼──────────────────────┤
│  Terminal C          │  Terminal D          │
│  Dashboard           │  Load generator /    │
│  python dashboard.py │  log watching        │
└──────────────────────┴──────────────────────┘
```

---

## Step-by-Step Instructions

### Terminal A — Cluster Setup & Monitoring

Run all commands in **Terminal A** in order:

```bash
# 1. Start Minikube
minikube start --driver=docker --cpus=4 --memory=8192
minikube addons enable metrics-server

# 2. Wait for metrics-server (~60 s)
kubectl top nodes

# 3. Point Docker CLI at Minikube's registry
eval $(minikube docker-env)

# 4. Build images
docker build -f Dockerfile.collector  -t cpu-collector:v1  .
docker build -f Dockerfile.autoscaler -t ml-autoscaler:v1  .
docker build -f kubernetes/Dockerfile -t cpu-load-app:v1   kubernetes/

# 5. Create namespace and deploy all resources
kubectl create namespace ai-scaler --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f kubernetes/autoscaler-rbac.yaml
kubectl apply -f kubernetes/collector.yaml
kubectl apply -f kubernetes/cpu-load-app.yaml
kubectl apply -f kubernetes/autoscaler-deployment.yaml

# 6. Wait until all pods are Running (~2 min)
kubectl get pods -n ai-scaler -w
```

Expected output once ready:
```
NAME                             READY   STATUS    RESTARTS
cpu-collector-xxx                1/1     Running   0
cpu-load-app-xxx                 1/1     Running   0
ml-autoscaler-xxx                1/1     Running   0
```

---

### Terminal B — Port-Forwards (KEEP THIS TERMINAL OPEN)

> ⚠️ **Critical**: these two commands must stay running the entire session.
> If they stop, the dashboard goes to "demo" mode and the autoscaler loses connectivity.

```bash
# Forward autoscaler API (5000) — run in background
kubectl port-forward -n ai-scaler deployment/ml-autoscaler 5000:5000 &

# Forward collector (8000)
kubectl port-forward -n ai-scaler svc/cpu-collector 8000:8000
```

Leave Terminal B open. If either port-forward dies, rerun the matching command.

---

### Terminal C — Dashboard

```bash
# From the repo root
python dashboard.py
```

Open **http://localhost:8080** in your browser.

- **● live** badge → port-forwards are active, receiving real data
- **● demo** badge → port-forward is down; restart Terminal B commands
- Cards update every 5 seconds

---

### Terminal D — Load Generation & Log Watching

Apply load to drive CPU up and trigger autoscaling:

```bash
# Option A — standard phased load (~8 min)
kubectl apply -f kubernetes/loadgen.yaml

# Option B — strong continuous load (restarts automatically)
kubectl apply -f kubernetes/loadgen-strong.yaml
```

Watch everything happen in real time:

```bash
# Pod count changes (open a sub-shell or new tab)
kubectl get pods -n ai-scaler -w

# ML autoscaler decisions
kubectl logs -f -n ai-scaler deployment/ml-autoscaler

# Collector output
kubectl logs -f -n ai-scaler deployment/cpu-collector
```

---

## Verifying the Metrics Pipeline

With port-forwards running (Terminal B):

```bash
# Raw / scaled / smoothed CPU from collector
curl http://localhost:8000/metrics/cpu | python -m json.tool

# Full autoscaler state (prediction, replicas, events)
curl http://localhost:5000/api/metrics | python -m json.tool
```

---

## ML vs HPA Comparison

### What Is HPA?

Kubernetes **Horizontal Pod Autoscaler (HPA)** is the built-in reactive scaler.
It watches CPU utilisation and adjusts replicas *after* a threshold is crossed.

| Property | Kubernetes HPA | ML Autoscaler (GRU) |
|----------|---------------|---------------------|
| Approach | Reactive | **Predictive** |
| Scaling trigger | CPU > threshold *now* | CPU predicted to be high *soon* |
| Scale-up latency | 15–60 s after overload | Scales **before** overload arrives |
| Scale-down | Fixed 5 min stabilisation window | Model-driven, avoids premature scale-down |
| Oscillation | Common under bursty load | Smoothed by prediction window |
| Cold-start awareness | None | Captured in training traces |

### Running the Side-by-Side Comparison

#### 1 — Deploy HPA

```bash
kubectl apply -f kubernetes/hpa-comparison.yaml
```

This creates a standard HPA targeting 50% CPU utilisation on `cpu-load-app`
(min 1 replica, max 10 replicas).

#### 2 — Verify HPA Is Active

```bash
kubectl get hpa -n ai-scaler
```

Expected output:
```
NAME               REFERENCE                 TARGETS        MINPODS   MAXPODS   REPLICAS
cpu-load-app-hpa   Deployment/cpu-load-app   <unknown>/50%  1         10        1
```

> `<unknown>` in the TARGETS column is normal for the first 30–60 seconds while the metrics server collects its initial CPU sample. It will change to a real percentage (e.g. `12%/50%`) once metrics are available.

#### 3 — Apply Load and Watch Both Scalers

Open two watch windows in Terminal D:

```bash
# Window 1 — pod count (shows HPA scaling)
watch -n2 kubectl get pods -n ai-scaler

# Window 2 — ML autoscaler log (shows GRU decisions)
kubectl logs -f -n ai-scaler deployment/ml-autoscaler
```

Then apply strong load:

```bash
kubectl apply -f kubernetes/loadgen-strong.yaml
```

#### 4 — Collect Comparison Metrics

```bash
# Replica history from HPA
kubectl describe hpa cpu-load-app-hpa -n ai-scaler

# ML autoscaler scaling events
curl http://localhost:5000/api/metrics | python -m json.tool | grep -A5 scaling_events
```

#### 5 — Expected Observations

| Observation | HPA | ML Autoscaler |
|-------------|-----|---------------|
| Time from load start to first scale-up | 30–90 s | < 30 s (predicts ahead) |
| Replica overshoot | Frequent | Rare |
| Stability during sustained load | Oscillates | Stable |
| Scale-down after load stops | After 5 min delay | Model-driven |

#### 6 — Remove HPA After Comparison

```bash
kubectl delete -f kubernetes/hpa-comparison.yaml
```

> ⚠️ Run only **one** scaler at a time in production. The ML autoscaler and HPA will fight each other if both are active and targeting the same deployment.

---

## Autoscaling Logic (ML Autoscaler)

The `ml-autoscaler` loop runs every 5 seconds:

1. `GET http://cpu-collector:8000/metrics/cpu` → `raw_cpu` (actual cores)
2. Linear map: `raw_cpu` in **0–2 cores** → **300–800** (model training range)
3. Maintain a **60-point sliding window**
4. Once 60 points collected → `gru_model.h5` predicts the next value
5. Decision table:

| Predicted CPU | Action |
|---------------|--------|
| > 0.5 | **Scale up** (+1 replica, max 10) |
| < 0.2 | **Scale down** (−1 replica, min 1) |
| 0.2 – 0.5 | **STABLE** — no change |

6. Scaling event logged to `/api/metrics` → dashboard table updates

---

## Troubleshooting

### Dashboard shows "● demo" instead of "● live"

Port-forward is not running. In Terminal B:
```bash
kubectl port-forward -n ai-scaler deployment/ml-autoscaler 5000:5000 &
kubectl port-forward -n ai-scaler svc/cpu-collector 8000:8000
```

### `ImagePullBackOff` on a pod

Docker CLI is not pointed at Minikube's registry. In Terminal A:
```bash
eval $(minikube docker-env)
docker build -f Dockerfile.autoscaler -t ml-autoscaler:v1 .
# (repeat for other images as needed)
kubectl rollout restart deployment/ml-autoscaler -n ai-scaler
```

### CPU shows `0.0` in dashboard

Metrics server is still warming up, or the collector cannot reach the Metrics API.
```bash
# Check metrics-server
kubectl top pods -n ai-scaler

# Check collector logs
kubectl logs -n ai-scaler deployment/cpu-collector
```
Wait 60–90 s and retry.

### Pods not scaling even with load

1. Confirm loadgen is running:
   ```bash
   kubectl get pods -n ai-scaler | grep loadgen
   kubectl logs -n ai-scaler pod/loadgen-strong
   ```
2. Check the autoscaler needs 60 data points before making predictions (~5 min after startup):
   ```bash
   curl http://localhost:5000/api/metrics | python -m json.tool | grep data_points
   ```
3. Confirm `loadgen-strong.yaml` uses `restartPolicy: Always` so load is continuous.

### `kubectl port-forward` dies immediately

The target pod may have restarted. Get the current pod name and retry:
```bash
kubectl get pods -n ai-scaler
kubectl port-forward -n ai-scaler pod/<exact-pod-name> 5000:5000
```

---

## Quick Reference Commands

```bash
# Status snapshot
kubectl get all -n ai-scaler

# Live pod watch
kubectl get pods -n ai-scaler -w

# ML autoscaler log stream
kubectl logs -f -n ai-scaler deployment/ml-autoscaler

# Collector log stream
kubectl logs -f -n ai-scaler deployment/cpu-collector

# Check HPA (if deployed)
kubectl get hpa -n ai-scaler
kubectl describe hpa cpu-load-app-hpa -n ai-scaler

# Force manual scale
kubectl scale deployment cpu-load-app --replicas=3 -n ai-scaler

# Delete loadgen (stop load)
kubectl delete pod loadgen        -n ai-scaler --ignore-not-found
kubectl delete pod loadgen-strong -n ai-scaler --ignore-not-found

# Full cleanup
kubectl delete namespace ai-scaler
minikube stop
```

---

## Model Performance

The GRU model was trained on Google cluster CPU traces:

| Metric | GRU (used) | LSTM (baseline) |
|--------|-----------|-----------------|
| R²     | **0.9138** | 0.8921 |
| RMSE   | **19.72**  | 22.45  |
| MAE    | **9.20**   | 11.83  |

---

## Pre-Demo Checklist

- [ ] `minikube status` shows **Running**
- [ ] `kubectl top nodes` returns CPU/memory values (metrics-server ready)
- [ ] All three pods in `ai-scaler` are **1/1 Running**
- [ ] Port-forwards in Terminal B are alive
- [ ] `curl http://localhost:5000/api/metrics` returns JSON
- [ ] Dashboard at `http://localhost:8080` shows **● live**
- [ ] Loadgen pod is running (check with `kubectl get pods -n ai-scaler`)
- [ ] `data_points` in `/api/metrics` is ≥ 60 (predictions active)
