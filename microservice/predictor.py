"""
predictor.py
Load the trained GRU model and make CPU predictions
"""

import numpy as np
import pickle
import os
from keras.models import load_model
from typing import List, Optional

class CPUPredictor:
    """Load trained GRU model and make predictions"""
    
    def __init__(self, model_path: str = "models/gru_model.h5", 
                 scaler_path: str = "models/scaler.pkl"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to GRU model file
            scaler_path: Path to MinMaxScaler pickle file
        """
        try:
            # Load trained GRU model
            self.model = load_model(model_path)
            print(f"✓ Loaded GRU model from {model_path}")
            
            # Load scaler (for data normalization)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded scaler from {scaler_path}")
            
            self.is_ready = True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            self.scaler = None
            self.is_ready = False
    
    def predict_next_cpu(self, recent_cpu_values: List[float]) -> Optional[float]:
        """
        Predict next CPU value
        
        Args:
            recent_cpu_values: Last 60 CPU values (scaled 300-800)
        
        Returns:
            Predicted CPU value (scaled 300-800), or None if prediction fails
        """
        if not self.is_ready or self.model is None:
            return None
        
        if len(recent_cpu_values) < 60:
            print(f"⚠ Need 60 values for prediction, have {len(recent_cpu_values)}")
            return None
        
        try:
            # Take last 60 values
            data = np.array(recent_cpu_values[-60:]).reshape(-1, 1)
            
            # Normalize using training scaler
            data_scaled = self.scaler.transform(data)
            
            # Reshape for model: (1, 60, 1)
            X = data_scaled.reshape(1, 60, 1)
            
            # Predict (returns normalized value)
            prediction_normalized = self.model.predict(X, verbose=0)[0][0]
            
            # Inverse transform back to original scale (300-800)
            prediction_original = self.scaler.inverse_transform(
                [[prediction_normalized]]
            )[0][0]
            
            return float(prediction_original)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """Get info about loaded model"""
        if not self.is_ready:
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "model_type": "GRU",
            "r2_score": 0.9138,
            "rmse": 19.7157,
            "mae": 9.1969
        }
