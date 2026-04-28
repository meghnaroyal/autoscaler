"""
api.py
FastAPI endpoints for metrics and predictions
"""

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from datetime import datetime
import asyncio
import os

from kubelet_client import KubeletClient
from smoother import ExponentialMovingAverageSmoother
from storage import MetricsStorage
from scaler import CPUScaler
from predictor import CPUPredictor

# Initialize FastAPI app
app = FastAPI(
    title="CPU Metrics Microservice",
    description="Collects absolute CPU values, scales them, and feeds to ML model",
    version="1.0.0"
)

# Initialize components
SIMULATE_MODE = False  # TURN OFF SIMULATION - USE REAL KUBELET
kubelet = KubeletClient(simulate=SIMULATE_MODE)
smoother = ExponentialMovingAverageSmoother(alpha=0.3)
scaler = CPUScaler(raw_min=0.0, raw_max=1.0, scaled_min=300.0, scaled_max=800.0)
storage = MetricsStorage(max_entries=1440)
predictor = CPUPredictor()

# Global state
current_raw_cpu = 0.0
current_scaled_cpu = 0.0
current_smoothed_cpu = 0.0
last_error = None


@app.get("/metrics/cpu", tags=["metrics"])
async def get_cpu_metrics():
    """Get current CPU metrics"""
    return {
        "raw_cpu": round(current_raw_cpu, 6),
        "scaled_cpu": round(current_scaled_cpu, 1),
        "smoothed_cpu": round(current_smoothed_cpu, 1),
        "timestamp": datetime.now().isoformat(),
        "unit": "cores"
    }


@app.get("/metrics/history", tags=["metrics"])
async def get_history(minutes: int = Query(60, ge=1, le=1440)):
    """Get historical CPU data (scaled values)"""
    history = storage.get_last_n_minutes(minutes)
    
    if not history:
        return {"data": []}
    
    timestamps = list(storage.get_all().keys())
    relevant_timestamps = timestamps[-len(history):]
    
    data = []
    for ts, entry in zip(relevant_timestamps, history):
        data.append({
            "cpu": round(entry["cpu"], 1),
            "timestamp": ts
        })
    
    return {"data": data}


@app.get("/metrics/averages", tags=["metrics"])
async def get_averages():
    """Get CPU averages for different time windows"""
    return {
        "current": round(current_scaled_cpu, 1),
        "avg_1m": round(storage.get_average(1), 1),
        "avg_5m": round(storage.get_average(5), 1),
        "avg_15m": round(storage.get_average(15), 1),
        "avg_60m": round(storage.get_average(60), 1),
        "unit": "cores (scaled)"
    }


@app.get("/metrics/statistics", tags=["metrics"])
async def get_statistics():
    """Get statistics about stored metrics"""
    stats = storage.get_statistics()
    return {
        "min": round(stats["min"], 1),
        "max": round(stats["max"], 1),
        "avg": round(stats["avg"], 1),
        "count": stats["count"],
        "unit": "cores (scaled)"
    }


@app.get("/predict/next-cpu", tags=["predictions"])
async def predict_next_cpu():
    """Predict next CPU value using GRU model"""
    history = storage.get_last_n_minutes(60)
    
    if len(history) < 60:
        return {
            "error": f"Need 60 data points, have {len(history)}",
            "data_points": len(history)
        }
    
    cpu_values = [h["cpu"] for h in history]
    prediction = predictor.predict_next_cpu(cpu_values)
    
    if prediction is None:
        return {"error": "Prediction failed"}
    
    return {
        "predicted_cpu": round(prediction, 1),
        "current_cpu": round(current_scaled_cpu, 1),
        "unit": "cores (scaled)",
        "confidence": "91.4% (R² = 0.9138)",
        "model": "GRU"
    }


@app.get("/predict/info", tags=["predictions"])
async def get_model_info():
    """Get trained model information"""
    return predictor.get_model_info()


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "simulate_mode": SIMULATE_MODE,
        "last_error": last_error
    }


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    print("\n" + "=" * 70)
    print("🚀 CPU METRICS MICROSERVICE STARTING")
    print("=" * 70)
    print(f"Simulate Mode: {SIMULATE_MODE}")
    print(f"Scaling: 0-1 cores (Kubelet) → 300-800 cores (training range)")
    print("=" * 70 + "\n")
    
    asyncio.create_task(collect_metrics_loop())


async def collect_metrics_loop():
    """Background task: collect metrics every 10 seconds"""
    global current_raw_cpu, current_scaled_cpu, current_smoothed_cpu, last_error
    
    print("✓ Starting metrics collection loop (10 second interval)...\n")
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            
            # STEP 1: Get raw CPU from Kubelet
            current_raw_cpu = kubelet.get_total_cpu_cores()
            
            # STEP 2: Scale to training range (300-800)
            current_scaled_cpu = scaler.scale(current_raw_cpu)
            
            # STEP 3: Smooth scaled values
            current_smoothed_cpu = smoother.smooth(current_scaled_cpu)
            
            # STEP 4: Store for history
            storage.store(current_scaled_cpu)
            
            # STEP 5: Print status
            print(f"[{iteration}] Raw: {current_raw_cpu:.6f} cores → "
                  f"Scaled: {current_scaled_cpu:6.1f} cores | "
                  f"Smoothed: {current_smoothed_cpu:6.1f} cores")
            
            last_error = None
            
        except Exception as e:
            error_msg = f"Error in collection loop: {e}"
            print(f"❌ {error_msg}")
            last_error = str(e)
        
        await asyncio.sleep(10)
