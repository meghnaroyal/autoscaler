import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

print("=" * 55)
print("  STEP 2 — TRAIN GRU MODEL")
print("=" * 55)

X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"Data loaded:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test : {X_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(GRU(64, input_shape=(60, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining GRU model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

model.save("models/gru_model.h5")
print("\nModel saved to models/gru_model.h5")

predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
y_actual    = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_actual, predictions))
mae  = mean_absolute_error(y_actual, predictions)
r2   = r2_score(y_actual, predictions)

print("\n" + "=" * 55)
print("  GRU EVALUATION METRICS")
print("=" * 55)
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")
print("=" * 55)

with open("results/gru_metrics.txt", "w") as f:
    f.write(f"GRU Model Evaluation\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE : {mae:.4f}\n")
    f.write(f"R2  : {r2:.4f}\n")

plt.figure(figsize=(12, 5))
plt.plot(y_actual[:200],    label="Actual CPU",    color="steelblue")
plt.plot(predictions[:200], label="GRU Predicted", color="coral", linestyle="--")
plt.title("GRU — Actual vs Predicted CPU")
plt.xlabel("Time step")
plt.ylabel("CPU Usage")
plt.legend()
plt.tight_layout()
plt.savefig("results/gru_predictions.png")
plt.close()
print("Plot saved to results/gru_predictions.png")
print("\nGRU training complete. Run 3_train_lstm.py next.")
