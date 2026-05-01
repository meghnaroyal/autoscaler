"""
kubelet_client.py
Fetches ABSOLUTE CPU values from Kubernetes Metrics API
"""

import requests
import os
import math
import random
import time
from typing import Dict, List

class KubeletClient:
    """
    Collects absolute CPU usage from Metrics API
    Can simulate realistic 0-10 core patterns for testing
    """
    
    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self.start_time = time.time()
        self.namespace = os.getenv("NAMESPACE", "ai-scaler")
        
        if simulate:
            print(f"✓ KubeletClient in SIMULATION mode (0-10 cores)")
            return
        
        # Real Metrics API mode
        self.api_host = "https://kubernetes.default.svc"
        self.api_path = f"/apis/metrics.k8s.io/v1beta1/namespaces/{self.namespace}/pods"
        
        self.session = requests.Session()
        
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                self.token = f.read().strip()
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        
        self.session.verify = False
        print(f"✓ KubeletClient initialized: Metrics API for namespace={self.namespace}")
    
    def get_total_cpu_cores(self) -> float:
        """Get total CPU in cores (0-10 range for Minikube)"""
        if self.simulate:
            return self._simulate_cpu()
        return self._get_real_cpu()
    
    def _simulate_cpu(self) -> float:
        """Simulate realistic 0-10 core patterns"""
        elapsed = time.time() - self.start_time
        
        base = 5.0
        wave = 3.0 * math.sin(2 * math.pi * elapsed / 300)
        noise = random.uniform(-0.5, 0.5)
        
        cpu = base + wave + noise
        cpu = max(0.1, min(10.0, cpu))
        
        return cpu
    
    def _get_real_cpu(self) -> float:
        """Get real CPU from Metrics API"""
        try:
            url = f"{self.api_host}{self.api_path}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            total_cpu_millicores = 0
            
            # Sum CPU from all pods in namespace
            for item in data.get('items', []):
                for container in item.get('containers', []):
                    cpu_str = container.get('usage', {}).get('cpu', '0')
                    
                    # Parse CPU string (e.g., "50m" -> 50 millicores, "500n" -> nanocores, "2u" -> microcores)
                    if cpu_str.endswith('n'):
                        # nanocores: 1,000,000 nanocores = 1 millicore
                        cpu_millicores = int(cpu_str[:-1]) / 1_000_000
                    elif cpu_str.endswith('u'):
                        # microcores: 1,000 microcores = 1 millicore
                        cpu_millicores = int(cpu_str[:-1]) / 1_000
                    elif cpu_str.endswith('m'):
                        cpu_millicores = int(cpu_str[:-1])
                    else:
                        cpu_millicores = int(float(cpu_str) * 1000)
                    
                    total_cpu_millicores += cpu_millicores
            
            total_cpu_cores = total_cpu_millicores / 1000.0
            return total_cpu_cores
            
        except Exception as e:
            print(f"❌ Error fetching Metrics API CPU: {e}")
            return 0.0
