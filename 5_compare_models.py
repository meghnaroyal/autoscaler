import numpy as np
import matplotlib.pyplot as plt
import pickle

print("=" * 55)
print("  STEP 5 — COMPARE MODELS")
print("=" * 55)

def read_metrics(filepath):
    metrics = {}
    with open(filepath, "r") as f:
        for line in f.readlines()[1:]:
            key, value = line.strip().split(": ")
            metrics[key.strip()] = float(value)
    return metrics

gru_m  = read_metrics("results/gru_metrics.txt")
lstm_m = read_metrics("results/lstm_metrics.txt")

print("\nModel Comparison:")
print(f"{'Metric':<10} {'GRU':>10} {'LSTM':>10}")
print("-" * 32)
print(f"{'RMSE':<10} {gru_m['RMSE']:>10.4f} {lstm_m['RMSE']:>10.4f}")
print(f"{'MAE':<10} {gru_m['MAE']:>10.4f} {lstm_m['MAE']:>10.4f}")
print(f"{'R2':<10} {gru_m['R2']:>10.4f} {lstm_m['R2']:>10.4f}")

models = ["GRU", "LSTM"]
rmses  = [gru_m["RMSE"], lstm_m["RMSE"]]
maes   = [gru_m["MAE"],  lstm_m["MAE"]]
r2s    = [gru_m["R2"],   lstm_m["R2"]]

best_rmse = models[rmses.index(min(rmses))]
best_r2 = models[r2s.index(max(r2s))]
print(f"\nBest model by RMSE: {best_rmse}")
print(f"Best model by R²: {best_r2}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(models, rmses, color=["steelblue", "green"])
axes[0].set_title("RMSE (lower is better)")
axes[0].set_ylabel("RMSE")

axes[1].bar(models, maes, color=["steelblue", "green"])
axes[1].set_title("MAE (lower is better)")
axes[1].set_ylabel("MAE")

axes[2].bar(models, r2s, color=["steelblue", "green"])
axes[2].set_title("R² Score (higher is better)")
axes[2].set_ylabel("R²")

plt.suptitle("Model Comparison — GRU vs LSTM", fontsize=14)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.close()
print("\nComparison chart saved to results/model_comparison.png")

with open("results/comparison_summary.txt", "w") as f:
    f.write("Model Comparison Summary\n")
    f.write("=" * 32 + "\n")
    f.write(f"{'Metric':<10} {'GRU':>10} {'LSTM':>10}\n")
    f.write("-" * 32 + "\n")
    f.write(f"{'RMSE':<10} {gru_m['RMSE']:>10.4f} {lstm_m['RMSE']:>10.4f}\n")
    f.write(f"{'MAE':<10} {gru_m['MAE']:>10.4f} {lstm_m['MAE']:>10.4f}\n")
    f.write(f"{'R2':<10} {gru_m['R2']:>10.4f} {lstm_m['R2']:>10.4f}\n")
    f.write(f"\nBest model by RMSE: {best_rmse}\n")
    f.write(f"Best model by R²: {best_r2}\n")

print("\n✅ Model comparison complete!")
