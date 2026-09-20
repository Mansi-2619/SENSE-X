from pathlib import Path
import sys

import numpy as np


# Add project root to Python path
BASE = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BASE)
)


from src.sensors.radar_model import predict_radar


# ============================================================
# LOAD ONE REAL TEST SAMPLE
# ============================================================

X_test = np.load(
    BASE
    / "data"
    / "processed"
    / "radar"
    / "X_test.npy"
)

y_test = np.load(
    BASE
    / "data"
    / "processed"
    / "radar"
    / "y_test.npy"
)


# Pick first test sample
sample_index = 0

sample = X_test[sample_index]

actual_label = int(
    y_test[sample_index]
)


# ============================================================
# PREDICTION
# ============================================================

result = predict_radar(sample)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("SENSE-X RADAR INFERENCE TEST")
print("=" * 60)

print(
    f"\nInput shape: {sample.shape}"
)

print(
    f"Actual label: "
    f"{actual_label}"
)

print(
    f"Predicted label: "
    f"{result['label']}"
)

print(
    f"Human probability: "
    f"{result['human_probability']:.4f}"
)

print(
    f"No-human probability: "
    f"{result['no_human_probability']:.4f}"
)

print(
    f"Prediction: "
    f"{result['prediction']}"
)

print("\n" + "=" * 60)
print("RADAR INFERENCE TEST COMPLETE")
print("=" * 60)