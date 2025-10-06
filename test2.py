# gravity_keras.py
# Learn Newton's gravity: F = G * M * m / r^2
# Functional Keras model using log-space linearization + robust Windows-safe printing.

import os
import sys
import math
import numpy as np

# --- Optional: force UTF-8 so fancy symbols won't crash on Windows (safe if unsupported) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+
except Exception:
    pass

# --- If you use an IDE/terminal that still isn't UTF-8, we fallback to ASCII symbols later ---

# Make sure TensorFlow is available
try:
    import tensorflow as tf
except ModuleNotFoundError as e:
    raise SystemExit(
        "TensorFlow is not installed. Install it with:\n"
        "  pip install tensorflow\n"
        "or (for CPU-only):\n"
        "  pip install tensorflow-cpu\n"
    ) from e

# -----------------------------
# 1) Reproducibility & constants
# -----------------------------
np.random.seed(42)
tf.random.set_seed(42)

G = 6.674e-11  # m^3 kg^-1 s^-2  (gravitational constant)
EPS = 1e-12    # to avoid log(0)

# -----------------------------
# 2) Synthetic dataset
# -----------------------------
def log_uniform(low, high, n, rng=None):
    """Sample log-uniform in [low, high]."""
    rng = rng or np.random
    return np.exp(rng.uniform(np.log(low), np.log(high), size=n))

N = 60_000  # total samples
# Choose broad ranges (adapt to your use-case)
M = log_uniform(1e20, 1e30, N)   # kg
m = log_uniform(1e3,  1e8,  N)   # kg
r = log_uniform(1,  1e9,  N)   # m (avoid tiny r to prevent blow-ups)

F = G * (M * m) / (r**2)         # Newton's law

# Features and targets
X = np.stack([M, m, r], axis=1).astype(np.float64)  # shape (N, 3)
y = F.astype(np.float64).reshape(-1, 1)             # shape (N, 1)

# Train/val/test split
idx = np.arange(N)
np.random.shuffle(idx)
train_end = int(0.8 * N)
val_end   = int(0.9 * N)

X_train, y_train = X[idx[:train_end]], y[idx[:train_end]]
X_val,   y_val   = X[idx[train_end:val_end]], y[idx[train_end:val_end]]
X_test,  y_test  = X[idx[val_end:]], y[idx[val_end:]]

# ---------------------------------------
# 3) Functional model in log-space (best)
#    log F = log G + log M + log m - 2 log r
# ---------------------------------------
inp = tf.keras.Input(shape=(3,), dtype=tf.float64, name="raw_features")  # [M, m, r]

log_feats = tf.keras.layers.Lambda(
    lambda t: tf.math.log(t + EPS), name="log_features", dtype=tf.float64
)(inp)  # -> [log M, log m, log r]

logF = tf.keras.layers.Dense(
    units=1, activation="linear", use_bias=True,
    kernel_initializer="zeros", bias_initializer="zeros",
    dtype=tf.float64, name="logF_linear"
)(log_feats)

F_pred = tf.keras.layers.Lambda(lambda t: tf.exp(t), name="F", dtype=tf.float64)(logF)

model = tf.keras.Model(inputs=inp, outputs=F_pred, name="GravityLogLinear")

# Log-space MSE (scale-invariant, matches multiplicative physics)
def log_mse(y_true, y_pred):
    return tf.reduce_mean(
        tf.square(tf.math.log(y_true + EPS) - tf.math.log(y_pred + EPS))
    )

model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=log_mse)

# ---------------------------------------
# 4) Train with early stopping
# ---------------------------------------
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=512, verbose=1, callbacks=[es]
)

# ---------------------------------------
# 5) Inspect learned coefficients
# ---------------------------------------
W, b = model.get_layer("logF_linear").get_weights()  # W:(3,1), b:(1,)
W = W.reshape(-1)        # [w_M, w_m, w_r]
b = float(b.reshape(())) # scalar

theoretical_W = np.array([1.0, 1.0, -2.0])
theoretical_b = math.log(G)

# Pick a safe approximation symbol based on encoding
approx_symbol = "≈"
try:
    "test".encode(sys.stdout.encoding or "utf-8")
except Exception:
    approx_symbol = "~="  # fallback if encoding is unknown

print("\n=== Learned vs Theoretical Coefficients ===")
print("Learned W (logM, logm, logr):", W)
print("Theoretical W:                [1.0, 1.0, -2.0]")
print(f"Learned bias {approx_symbol} log(G):  {b:.10f}")
print(f"Theoretical log(G):           {theoretical_b:.10f}")

# ---------------------------------------
# 6) Evaluate on test data
# ---------------------------------------
y_pred = model.predict(X_test, verbose=0)

rel_err = np.abs((y_pred - y_test) / (y_test + EPS))
mean_rel_err = float(np.mean(rel_err))
p95_rel_err  = float(np.percentile(rel_err, 95))
p99_rel_err  = float(np.percentile(rel_err, 99))

print("\n=== Test Performance (relative errors) ===")
print(f"Mean relative error:          {mean_rel_err:.6e}")
print(f"95th percentile relative err: {p95_rel_err:.6e}")
print(f"99th percentile relative err: {p99_rel_err:.6e}")
print(f"Final val log-MSE:            {history.history['val_loss'][-1]:.6e}")

# ---------------------------------------
# 7) (Optional) Save the model
# ---------------------------------------
# model.save("gravity_loglinear_keras_model.keras")
# print("\nSaved model to gravity_loglinear_keras_model.keras")
