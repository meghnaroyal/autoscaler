# ML Autoscaler — End-to-End Setup Guide

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Minikube Cluster (namespace: ai-scaler)                │
│                                                         │
│  ┌──────────────┐   CPU metrics   ┌──────────────────┐  │
│  │ cpu-load-app │ ←── loadgen ──→ │  cpu-collector   │  │
│  │  (workload)  │                 │  :8000           │  │
│  └──────────────┘                 └────────┬─────────┘  │
│                                            │ raw_cpu    │
│                                   ┌────────▼─────────┐  │
│                                   │  ml-autoscaler   │  │
│                                   │  :5000 /api/     │  │
│                                   │  (GRU model)     │  │
│                                   └────────┬─────────┘  │
└────────────────────────────────────────────┼────────────┘
                                             │ port-forward
                             ┌───────────────▼────────────┐
                             │  dashboard.py  :8080        │
                             │  http://localhost:8080      │
                             └────────────────────────────┘
```

**Four moving parts:**
| Service | Port | Purpose |
|---------|------|---------|
| `cpu-collector` | 8000 | Reads real CPU from Kubernetes Metrics API, scales 0–2 cores → 300–800 |
| `ml-autoscaler` | 5000 | Polls collector, runs GRU predictions, scales `cpu-load-app` replicas |
| `dashboard.py` | 8080 | Flask UI — polls autoscaler `/api/metrics`, auto-refreshes every 5 s |
| `cpu-load-app` | 8000 | Target workload; `loadgen` pods hammer its `/work` endpoint |

---

## Prerequisites (install once)

```bash
# 1. Minikube
brew install minikube            # macOS
# or follow https://minikube.sigs.k8s.io/docs/start/ for Linux/Windows

# 2. kubectl  (usually bundled with Docker Desktop / minikube)
brew install kubectl

# 3. Docker Desktop (or Docker Engine on Linux)

# 4. Python 3.10+
python3 --version

# 5. Python packages for running dashboard/autoscaler locally
pip install flask requests tensorflow keras scikit-learn numpy kubernetes
```

---

## Step 1 — Start Minikube

```bash
minikube start --driver=docker --cpus=4 --memory=8192
minikube addons enable metrics-server

# Verify metrics-server is ready (takes ~60 s)
kubectl top nodes
```

> **VS Code**: run task **"1 · Minikube: Start"** via `Ctrl+Shift+P → Tasks: Run Task`.

---

## Step 2 — Build Docker Images Inside Minikube

Point your local Docker CLI at Minikube's internal registry so images are
available to pods without pushing to Docker Hub:

```bash
eval $(minikube docker-env)       # Linux/macOS
# Windows PowerShell: & minikube -p minikube docker-env | Invoke-Expression

docker build -f Dockerfile.collector  -t cpu-collector:v1  .
docker build -f Dockerfile.autoscaler -t ml-autoscaler:v1  .
docker build -f kubernetes/Dockerfile -t cpu-load-app:v1   kubernetes/
```

> **VS Code**: run task **"3 · Docker: Build ALL images"**.

---

## Step 3 — Deploy to Minikube

```bash
# Create namespace (idempotent)
kubectl create namespace ai-scaler --dry-run=client -o yaml | kubectl apply -f -

# Apply manifests
kubectl apply -f kubernetes/autoscaler-rbac.yaml
kubectl apply -f kubernetes/collector.yaml
kubectl apply -f kubernetes/cpu-load-app.yaml
kubectl apply -f kubernetes/autoscaler-deployment.yaml

# Wait for all pods to reach Running state
kubectl get pods -n ai-scaler -w
```

Expected output (after ~2 min):
```
NAME                             READY   STATUS    RESTARTS
cpu-collector-xxx                1/1     Running   0
cpu-load-app-xxx                 1/1     Running   0
ml-autoscaler-xxx                1/1     Running   0
```

> **VS Code**: tasks **"4 · kubectl: Deploy ALL"** then **"5 · kubectl: Watch pods"**.

---

## Step 4 — Expose Services Locally (port-forward)

Open two terminals and leave them running:

```bash
# Terminal A — autoscaler metrics API
kubectl port-forward -n ai-scaler deployment/ml-autoscaler 5000:5000

# Terminal B — collector (optional, for manual inspection)
kubectl port-forward -n ai-scaler svc/cpu-collector 8000:8000
```

> **VS Code**: tasks **"6 · Port-forward: Autoscaler API (5000)"** and
> **"6 · Port-forward: CPU Collector (8000)"**.

---

## Step 5 — Run the Dashboard

```bash
# New terminal from the repo root
python dashboard.py
```

Open **http://localhost:8080** in a browser.

- The badge shows **● live** once port-forward is active (● demo data otherwise)
- Auto-refreshes every 5 seconds
- Cards: Current CPU · ML Model · Scaled CPU Recommendation · Current Pods · Pods Added
- Table: Recent Scaling Events

> **VS Code**: press **F5** and pick `"Dashboard (port 8080)"`.

---

## Step 6 — Verify the Metrics Pipeline

With the collector port-forwarded on 8000:

```bash
# Current raw / scaled / smoothed CPU
curl http://localhost:8000/metrics/cpu

# GRU prediction (requires 60 data points — wait ~10 min after step 3)
curl http://localhost:8000/predict/next-cpu

# Autoscaler state
curl http://localhost:5000/api/metrics | python -m json.tool
```

---

## Step 7 — Run Load Tests

Apply the load generator to drive CPU up and trigger autoscaling:

```bash
# Standard load: low → medium → spike → cooldown (~8 min)
kubectl apply -f kubernetes/loadgen.yaml

# OR strong load: medium → high → cooldown (~6 min, more aggressive)
kubectl apply -f kubernetes/loadgen-strong.yaml
```

Watch it happen in real time:

```bash
# Pod count changes
kubectl get pods -n ai-scaler -w

# Autoscaler decision log
kubectl logs -f -n ai-scaler deployment/ml-autoscaler

# Loadgen progress
kubectl logs -f -n ai-scaler pod/loadgen
```

> **VS Code**: tasks **"7 · Load test: Standard"** / **"7 · Load test: Strong"** and
> **"8 · Logs: Autoscaler"**.

---

## Autoscaling Logic

The `ml-autoscaler` loop (every 5 s):

1. `GET http://cpu-collector:8000/metrics/cpu` → `raw_cpu` (actual cores)
2. Scale: `raw_cpu` in **0–2 cores** → **model range 300–800** via linear mapping
3. Maintain a 60-point sliding window
4. Once 60 points are collected → `gru_model.h5` predicts the next value
5. Decision:
   - `predicted_cpu > 0.5` → **scale up** (add 1 replica, max 5)
   - `predicted_cpu < 0.2` → **scale down** (remove 1 replica, min 1)
   - Otherwise → **STABLE**
6. Scaling event is written to the `/api/metrics` response → dashboard updates

---

## Running Locally Without Kubernetes (Simulation Mode)

Useful for UI development without a running cluster:

```bash
# Terminal 1 – collector in simulation mode (generates synthetic CPU waves)
cd microservice
SIMULATE_MODE=true python main.py

# Terminal 2 – autoscaler pointing at local collector
COLLECTOR_URL=http://localhost:8000/metrics/cpu python autoscaler_controller.py

# Terminal 3 – dashboard
python dashboard.py
```

> **VS Code**: press **F5** and pick `"Full Stack (Collector + Autoscaler + Dashboard)"`.
> This runs all three processes simultaneously with simulation enabled.

---

## Confirming the Model Works

During a load test, watch the dashboard **Scaled CPU Recommendation** card:

- It should rise **before** the **Current CPU** card peaks — this is the GRU predicting future demand
- Scale-up events appear in the **Recent Scaling Events** table with timestamps
- After the loadgen finishes, CPU drops and the autoscaler automatically scales back down

Model performance (trained on Google cluster traces):

| Metric | GRU | LSTM |
|--------|-----|------|
| R²     | 0.9138 | 0.8921 |
| RMSE   | 19.72  | 22.45  |
| MAE    | 9.20   | 11.83  |

---

## Cleanup

```bash
kubectl delete namespace ai-scaler   # removes all resources
minikube stop                         # stop the cluster (keeps data)
minikube delete                       # completely remove the VM
```

> **VS Code**: task **"9 · Cleanup: Delete all ai-scaler resources"**.
