"""
smoother.py
Exponential Moving Average (EMA) to remove spikes
"""

class ExponentialMovingAverageSmoother:
    """Smooths spiky CPU data using EMA"""
    
    def __init__(self, alpha: float = 0.3):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be 0-1, got {alpha}")
        self.alpha = alpha
        self.prev_ema = None
    
    def smooth(self, current_value: float) -> float:
        """Apply EMA smoothing"""
        if self.prev_ema is None:
            self.prev_ema = current_value
            return current_value
        
        ema = (self.alpha * current_value + (1 - self.alpha) * self.prev_ema)
        self.prev_ema = ema
        return ema
    
    def reset(self):
        self.prev_ema = None
