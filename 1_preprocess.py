import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

print("=" * 55)
print("  STEP 1 — DATA PREPROCESSING")
print("=" * 55)

# Load data
df = pd.read_csv("data/processed_cpu.csv")
df["cpu_smoothed"] = df["cpu_usage"].rolling(window=5, center=True, min_periods=1).mean()
print(f"\nDataset loaded: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Use smoothed CPU values (absolute, not rate of change)
print("\n" + "=" * 55)
print("  CPU SMOOTHED STATISTICS")
print("=" * 55)
print(df["cpu_smoothed"].describe())

# Visualize
plt.figure(figsize=(12, 5))
plt.plot(df["cpu_smoothed"].values, color="steelblue", linewidth=0.8)
plt.title("CPU Smoothed Over Time (Google Cluster Trace)")
plt.xlabel("Time step (minutes)")
plt.ylabel("CPU Usage (cores)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cpu_timeseries.png", dpi=100)
plt.close()
print("\nPlot saved to results/cpu_timeseries.png")

# Normalize data
scaler = MinMaxScaler()
cpu_values = df["cpu_smoothed"].values.reshape(-1, 1)
cpu_scaled = scaler.fit_transform(cpu_values)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved to models/scaler.pkl")

# Create sliding window dataset
WINDOW_SIZE = 60

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X, y = create_dataset(cpu_scaled, WINDOW_SIZE)

print("\n" + "=" * 55)
print("  SLIDING WINDOW DATASET")
print("=" * 55)
print(f"X shape: {X.shape}  (samples, window, features)")
print(f"y shape: {y.shape}  (samples, target)")
print(f"Total samples: {len(X)}")

# Train-test split (80-20)
split = int(len(X) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Save for model training
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("\n" + "=" * 55)
print("  TRAIN-TEST SPLIT (80-20)")
print("=" * 55)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Total            : {len(X_train) + len(X_test)}")

print("\n" + "=" * 55)
print("✅ Preprocessing complete!")
print("=" * 55)
print("\nNext steps:")
print("  python3 2_train_gru.py")
print("  python3 3_train_lstm.py")
print("  python3 4_train_ar.py")
