"""
scaler.py
Scale CPU from Kubernetes Kubelet (actual range) to training data range (300-800 cores)
"""

class CPUScaler:
    """Scale CPU values to match training data range"""
    
    def __init__(self, 
                 raw_min: float = 0.0,
                 raw_max: float = 1.0,
                 scaled_min: float = 300.0,
                 scaled_max: float = 800.0):
        """
        Initialize scaler
        
        Args:
            raw_min: Min CPU from Kubelet (0 cores)
            raw_max: Max CPU from Kubelet (1 core - actual observed max)
            scaled_min: Min target (300 cores - training data range)
            scaled_max: Max target (800 cores - training data range)
        """
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.scaled_min = scaled_min
        self.scaled_max = scaled_max
        
        self.raw_range = raw_max - raw_min
        self.scaled_range = scaled_max - scaled_min
    
    def scale(self, raw_cpu: float) -> float:
        """
        Scale: 0-1 cores → 300-800 cores
        
        Example:
            scaler.scale(0.0)   → 300
            scaler.scale(0.5)   → 550
            scaler.scale(1.0)   → 800
        """
        raw_cpu = max(self.raw_min, min(self.raw_max, raw_cpu))
        
        normalized = (raw_cpu - self.raw_min) / self.raw_range
        scaled = normalized * self.scaled_range + self.scaled_min
        
        return scaled
    
    def unscale(self, scaled_cpu: float) -> float:
        """Reverse: 300-800 cores → 0-1 cores"""
        scaled_cpu = max(self.scaled_min, min(self.scaled_max, scaled_cpu))
        
        normalized = (scaled_cpu - self.scaled_min) / self.scaled_range
        raw = normalized * self.raw_range + self.raw_min
        
        return raw
