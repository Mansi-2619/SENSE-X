from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

DATA_DIR = BASE / "data" / "processed" / "radar"
MODEL_PATH = BASE / "models" / "radar" / "radar_human_detector.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# EXACT ARCHITECTURE USED DURING TRAINING
# ============================================================

class RadarHumanDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 4 -> 32
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 32 -> 64
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 64 -> 128
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 128 -> 128
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions, targets):

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    tp = np.sum(
        (predictions == 1) &
        (targets == 1)
    )

    tn = np.sum(
        (predictions == 0) &
        (targets == 0)
    )

    fp = np.sum(
        (predictions == 1) &
        (targets == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (targets == 1)
    )

    accuracy = (
        (tp + tn) / max(len(targets), 1)
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-8)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SENSE-X RADAR MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

X_test = np.load(
    DATA_DIR / "X_test.npy"
).astype(np.float32)

y_test = np.load(
    DATA_DIR / "y_test.npy"
).astype(np.int64)

print(f"Test data: {X_test.shape}")
print(f"Test labels: {y_test.shape}")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print(f"\nLoading model:")
print(MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = RadarHumanDetector().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

X_tensor = torch.from_numpy(X_test)

predictions = []
probabilities = []

batch_size = 256

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        batch_size
    ):

        end = start + batch_size

        X_batch = X_tensor[
            start:end
        ].to(DEVICE)

        logits = model(X_batch)

        probs = torch.softmax(
            logits,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )


predictions = np.asarray(predictions)
probabilities = np.asarray(probabilities)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

metrics = calculate_metrics(
    predictions,
    y_test
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"\nAccuracy:  {metrics['accuracy']:.4f}"
)

print(
    f"Precision: {metrics['precision']:.4f}"
)

print(
    f"Recall:    {metrics['recall']:.4f}"
)

print(
    f"F1 Score:  {metrics['f1']:.4f}"
)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

print("\nConfusion Matrix:")
print()
print("                 Predicted")
print("              No Human  Human")
print(
    f"No Human     {metrics['tn']:8d} "
    f"{metrics['fp']:6d}"
)
print(
    f"Human        {metrics['fn']:8d} "
    f"{metrics['tp']:6d}"
)

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print("\nClass distribution:")

print(
    f"Actual no-human: "
    f"{np.sum(y_test == 0)}"
)

print(
    f"Actual human:    "
    f"{np.sum(y_test == 1)}"
)

print(
    f"Predicted no-human: "
    f"{np.sum(predictions == 0)}"
)

print(
    f"Predicted human:    "
    f"{np.sum(predictions == 1)}"
)

# ------------------------------------------------------------
# Sample predictions
# ------------------------------------------------------------

print("\nSample predictions:")

for i in range(min(10, len(X_test))):

    predicted_class = predictions[i]

    human_probability = probabilities[i][1]

    actual_class = y_test[i]

    print(
        f"Sample {i:02d} | "
        f"Actual: {actual_class} | "
        f"Predicted: {predicted_class} | "
        f"Human probability: "
        f"{human_probability:.4f}"
    )

print("\n" + "=" * 70)
print("RADAR EVALUATION COMPLETE")
print("=" * 70)