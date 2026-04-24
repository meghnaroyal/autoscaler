"""
autoscaler_controller.py
ML-based autoscaler with Flask metrics API for dashboard
"""

import sys
import os
import numpy as np
import pickle
import requests
import time
import json
from datetime import datetime
from collections import deque
from threading import Thread
from flask import Flask, jsonify, render_template_string
from tensorflow.keras.models import load_model
from kubernetes import client, config

# Initialize Flask
app = Flask(__name__)

# Log to file
class DualLogger:
    def __init__(self, filename):
        self.file = open(filename, 'a')
        self.stdout = sys.stdout
    
    def write(self, msg):
        self.stdout.write(msg)
        self.stdout.flush()
        self.file.write(msg)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = DualLogger('/tmp/autoscaler.log')
sys.stderr = sys.stdout

print("="*60)
print("ML AUTOSCALER WITH DASHBOARD API STARTING")
print("="*60)

class MLAutoscaler:
    def __init__(self, model_path="models/gru_model.h5", scaler_path="models/scaler.pkl"):
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded!")
        
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded!")
        print(f"Scaler range: [{self.scaler.data_min_[0]:.2f}, {self.scaler.data_max_[0]:.2f}]")
        
        # Kubernetes setup
        print("Initializing Kubernetes client...")
        try:
            config.load_incluster_config()
            print("Using in-cluster config")
        except Exception as e:
            print(f"In-cluster config failed: {e}, trying kube config")
            config.load_kube_config()
        
        self.v1 = client.AppsV1Api()
        self.namespace = "ai-scaler"
        self.deployment_name = "cpu-load-app"
        self.collector_url = "http://cpu-collector:8000/metrics/cpu"
        
        # Config
        self.history_size = 60
        self.cpu_history = []
        self.scale_threshold_high = 0.5
        self.scale_threshold_low = 0.2
        self.max_replicas = 5
        self.min_replicas = 1
        
        # CPU scaling
        self.cpu_min_cores = 0.0
        self.cpu_max_cores = 2.0
        self.model_min = self.scaler.data_min_[0]
        self.model_max = self.scaler.data_max_[0]
        
        # Dashboard data
        self.current_cpu = 0.0
        self.predicted_cpu = 0.0
        self.current_replicas = 1
        self.status = "INITIALIZING"
        self.cpu_history_dashboard = deque(maxlen=120)  # Last 2 hours
        self.replica_history = deque(maxlen=120)
        self.scaling_events = deque(maxlen=50)
        
        print("✅ MLAutoscaler initialized")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Namespace: {self.namespace}")
        print(f"   Deployment: {self.deployment_name}")
        print(f"   CPU scaling: {self.cpu_min_cores}-{self.cpu_max_cores} cores → {self.model_min}-{self.model_max} model units")
        print(f"   Dashboard API: http://localhost:5000/api/metrics")
    
    def scale_cpu_value(self, cpu_cores):
        """Scale CPU value from cores to model range"""
        normalized = (cpu_cores - self.cpu_min_cores) / (self.cpu_max_cores - self.cpu_min_cores)
        normalized = np.clip(normalized, 0, 1)
        scaled = self.model_min + normalized * (self.model_max - self.model_min)
        return scaled
    
    def fetch_current_cpu(self):
        """Fetch current CPU from collector"""
        try:
            response = requests.get(self.collector_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("current_cpu", 0.0)
        except Exception as e:
            print(f"❌ Error fetching CPU: {e}")
            return 0.0
    
    def update_history(self, cpu_value):
        """Maintain sliding window of CPU values"""
        self.cpu_history.append(cpu_value)
        if len(self.cpu_history) > self.history_size:
            self.cpu_history.pop(0)
    
    def predict_next_cpu(self):
        """Use GRU model to predict next CPU"""
        if len(self.cpu_history) < self.history_size:
            self.status = f"COLLECTING_DATA ({len(self.cpu_history)}/60)"
            return None
        
        try:
            X = np.array(self.cpu_history).reshape(1, self.history_size, 1)
            X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(1, self.history_size, 1)
            
            prediction_scaled = self.model.predict(X_scaled, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
            
            normalized = (prediction - self.model_min) / (self.model_max - self.model_min)
            normalized = np.clip(normalized, 0, 1)
            prediction_cores = self.cpu_min_cores + normalized * (self.cpu_max_cores - self.cpu_min_cores)
            
            return max(0.0, prediction_cores)
        except Exception as e:
            print(f"❌ Error predicting: {e}")
            return None
    
    def get_current_replicas(self):
        """Get current replica count"""
        try:
            deployment = self.v1.read_namespaced_deployment(
                self.deployment_name, self.namespace
            )
            return deployment.spec.replicas
        except Exception as e:
            print(f"❌ Error reading deployment: {e}")
            return 1
    
    def scale_deployment(self, replicas):
        """Scale deployment to desired replicas"""
        try:
            deployment = self.v1.read_namespaced_deployment(
                self.deployment_name, self.namespace
            )
            deployment.spec.replicas = replicas
            self.v1.patch_namespaced_deployment(
                self.deployment_name, self.namespace, deployment
            )
            print(f"✅ Scaled {self.deployment_name} to {replicas} replicas")
            
            # Record event
            event = {
                "timestamp": datetime.now().isoformat(),
                "action": f"Scale to {replicas} replicas",
                "old_replicas": self.current_replicas,
                "new_replicas": replicas
            }
            self.scaling_events.appendleft(event)
            return True
        except Exception as e:
            print(f"❌ Error scaling deployment: {e}")
            return False
    
    def make_scaling_decision(self, predicted_cpu):
        """Decide scaling action based on prediction"""
        current_replicas = self.get_current_replicas()
        self.current_replicas = current_replicas
        
        print(f"\n📊 Prediction: {predicted_cpu:.4f} cores")
        print(f"   Current replicas: {current_replicas}")
        
        if predicted_cpu > self.scale_threshold_high:
            desired_replicas = min(current_replicas + 1, self.max_replicas)
            if desired_replicas > current_replicas:
                print(f"📈 SCALE UP: Predicted CPU {predicted_cpu:.4f} > {self.scale_threshold_high}")
                self.scale_deployment(desired_replicas)
                self.status = f"SCALING_UP ({current_replicas}→{desired_replicas})"
        
        elif predicted_cpu < self.scale_threshold_low:
            desired_replicas = max(current_replicas - 1, self.min_replicas)
            if desired_replicas < current_replicas:
                print(f"📉 SCALE DOWN: Predicted CPU {predicted_cpu:.4f} < {self.scale_threshold_low}")
                self.scale_deployment(desired_replicas)
                self.status = f"SCALING_DOWN ({current_replicas}→{desired_replicas})"
        
        else:
            print(f"➡️  HOLD: Predicted CPU within safe range")
            self.status = "STABLE"
    
    def get_metrics(self):
        """Return current metrics for dashboard"""
        return {
            "current_cpu": round(self.current_cpu, 4),
            "predicted_cpu": round(self.predicted_cpu, 4),
            "current_replicas": self.current_replicas,
            "status": self.status,
            "cpu_history": list(self.cpu_history_dashboard),
            "replica_history": list(self.replica_history),
            "scaling_events": list(self.scaling_events),
            "timestamp": datetime.now().isoformat()
        }
    
    def run(self, interval=5):
        """Main loop"""
        print(f"\n🚀 Starting autoscaler loop (interval={interval}s)")
        try:
            while True:
                cpu_cores = self.fetch_current_cpu()
                cpu_scaled = self.scale_cpu_value(cpu_cores)
                self.update_history(cpu_scaled)
                self.current_cpu = cpu_cores
                
                # Record for dashboard
                self.cpu_history_dashboard.append({
                    "timestamp": datetime.now().isoformat(),
                    "cpu": round(cpu_cores, 4)
                })
                self.replica_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "replicas": self.current_replicas
                })
                
                print(f"⏰ {time.strftime('%H:%M:%S')} | Current CPU: {cpu_cores:.4f} cores → {cpu_scaled:.2f} model units")
                
                prediction = self.predict_next_cpu()
                if prediction is not None:
                    self.predicted_cpu = prediction
                    self.make_scaling_decision(prediction)
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n✋ Autoscaler stopped")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()

# Global autoscaler instance
autoscaler = None

# Flask Routes
@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def api_metrics():
    """Return current metrics as JSON"""
    if autoscaler:
        return jsonify(autoscaler.get_metrics())
    return jsonify({"error": "Autoscaler not initialized"}), 500

def start_autoscaler():
    """Start autoscaler in a separate thread"""
    global autoscaler
    autoscaler = MLAutoscaler()
    autoscaler.run(interval=5)

# HTML Dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Autoscaler Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .metric-unit {
            color: #999;
            font-size: 0.9em;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .status-stable {
            background: #d4edda;
            color: #155724;
        }
        
        .status-scaling {
            background: #fff3cd;
            color: #856404;
        }
        
        .status-collecting {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .chart-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        
        .events-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .events-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        
        .event {
            padding: 15px;
            border-left: 4px solid #667eea;
            background: #f8f9ff;
            margin-bottom: 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .event:hover {
            background: #f0f2ff;
            transform: translateX(5px);
        }
        
        .event-time {
            font-size: 0.85em;
            color: #999;
        }
        
        .event-action {
            font-weight: bold;
            color: #667eea;
            margin-top: 5px;
        }
        
        .loading {
            text-align: center;
            color: white;
            font-size: 1.2em;
            padding: 20px;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 ML Autoscaler Dashboard</h1>
            <p class="subtitle">Real-time Kubernetes Resource Management</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Current CPU</div>
                <div class="metric-value" id="current-cpu">-</div>
                <div class="metric-unit">cores</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Predicted CPU</div>
                <div class="metric-value" id="predicted-cpu">-</div>
                <div class="metric-unit">cores</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Current Replicas</div>
                <div class="metric-value" id="current-replicas">-</div>
                <div class="metric-unit">pods</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="status-badge" id="status-badge">INITIALIZING</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">📈 CPU Usage (Last 2 Hours)</div>
                <canvas id="cpuChart"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">📊 Replica Count Timeline</div>
                <canvas id="replicaChart"></canvas>
            </div>
        </div>
        
        <div class="events-container">
            <div class="events-title">⚡ Recent Scaling Events</div>
            <div id="events-list">
                <div class="loading pulse">Loading events...</div>
            </div>
        </div>
    </div>
    
    <script>
        let cpuChart, replicaChart;
        
        function initCharts() {
            const ctx1 = document.getElementById('cpuChart').getContext('2d');
            cpuChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage (cores)',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 2,
                        pointBackgroundColor: '#667eea'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: true }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 2.0
                        }
                    }
                }
            });
            
            const ctx2 = document.getElementById('replicaChart').getContext('2d');
            replicaChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Replica Count',
                        data: [],
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 2,
                        pointBackgroundColor: '#764ba2'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: true }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            min: 0,
                            max: 5,
                            ticks: { stepSize: 1 }
                        }
                    }
                }
            });
        }
        
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    document.getElementById('current-cpu').textContent = data.current_cpu.toFixed(4);
                    document.getElementById('predicted-cpu').textContent = data.predicted_cpu.toFixed(4);
                    document.getElementById('current-replicas').textContent = data.current_replicas;
                    
                    // Update status
                    const statusBadge = document.getElementById('status-badge');
                    statusBadge.textContent = data.status;
                    statusBadge.className = 'status-badge';
                    if (data.status.includes('SCALING')) {
                        statusBadge.classList.add('status-scaling');
                    } else if (data.status.includes('COLLECTING')) {
                        statusBadge.classList.add('status-collecting');
                    } else {
                        statusBadge.classList.add('status-stable');
                    }
                    
                    // Update CPU chart
                    cpuChart.data.labels = data.cpu_history.map(d => 
                        new Date(d.timestamp).toLocaleTimeString()
                    );
                    cpuChart.data.datasets[0].data = data.cpu_history.map(d => d.cpu);
                    cpuChart.update('none');
                    
                    // Update replica chart
                    replicaChart.data.labels = data.replica_history.map(d => 
                        new Date(d.timestamp).toLocaleTimeString()
                    );
                    replicaChart.data.datasets[0].data = data.replica_history.map(d => d.replicas);
                    replicaChart.update('none');
                    
                    // Update events
                    const eventsList = document.getElementById('events-list');
                    if (data.scaling_events.length === 0) {
                        eventsList.innerHTML = '<div class="loading">No scaling events yet...</div>';
                    } else {
                        eventsList.innerHTML = data.scaling_events.map(event => `
                            <div class="event">
                                <div class="event-time">${new Date(event.timestamp).toLocaleString()}</div>
                                <div class="event-action">${event.action}</div>
                            </div>
                        `).join('');
                    }
                })
                .catch(error => console.error('Error fetching metrics:', error));
        }
        
        // Initialize
        initCharts();
        updateDashboard();
        
        // Update every 5 seconds
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    # Start autoscaler in background thread
    autoscaler_thread = Thread(target=start_autoscaler, daemon=True)
    autoscaler_thread.start()
    
    # Start Flask app
    print("\n" + "="*60)
    print("🌐 DASHBOARD AVAILABLE AT: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
