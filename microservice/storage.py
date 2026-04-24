"""
storage.py
In-memory storage for historical metrics
Keeps last 24 hours of CPU data (1440 minutes)
"""

from datetime import datetime
from collections import OrderedDict
from typing import Dict, List

class MetricsStorage:
    """Store CPU metrics with timestamps"""
    
    def __init__(self, max_entries: int = 1440):
        self.max_entries = max_entries
        self.data = OrderedDict()
    
    def store(self, cpu_cores: float, timestamp: datetime = None):
        """Store CPU value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        key = timestamp.isoformat()
        self.data[key] = {"cpu": cpu_cores}
        
        if len(self.data) > self.max_entries:
            self.data.popitem(last=False)
    
    def get_last_n_minutes(self, n: int) -> List[Dict]:
        """Get last N minutes of data"""
        if len(self.data) == 0:
            return []
        
        values = list(self.data.values())
        return values[-n:] if n > 0 else []
    
    def get_average(self, minutes: int = 1) -> float:
        """Get average CPU over N minutes"""
        data = self.get_last_n_minutes(minutes)
        
        if not data:
            return 0.0
        
        total = sum(d["cpu"] for d in data)
        return total / len(data)
    
    def get_all(self) -> Dict:
        """Get all stored data"""
        return dict(self.data)
    
    def get_statistics(self) -> Dict:
        """Get min, max, avg, count"""
        if not self.data:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}
        
        values = [d["cpu"] for d in self.data.values()]
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }
    
    def clear(self):
        self.data.clear()
