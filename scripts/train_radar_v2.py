from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

DATA_DIR = BASE / "data" / "processed" / "radar_v2"
MODEL_DIR = BASE / "models" / "radar_v2"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SENSE-X RADAR HUMAN DETECTOR")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

X_train = np.load(
    DATA_DIR / "X_train.npy"
).astype(np.float32)

y_train = np.load(
    DATA_DIR / "y_train.npy"
).astype(np.int64)

X_val = np.load(
    DATA_DIR / "X_val.npy"
).astype(np.float32)

y_val = np.load(
    DATA_DIR / "y_val.npy"
).astype(np.int64)

X_test = np.load(
    DATA_DIR / "X_test.npy"
).astype(np.float32)

y_test = np.load(
    DATA_DIR / "y_test.npy"
).astype(np.int64)


print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# MODEL
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

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                64,
                2,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = RadarHumanDetector().to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets,
):

    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )

    tp = np.sum(
        (predictions == 1)
        & (targets == 1)
    )

    tn = np.sum(
        (predictions == 0)
        & (targets == 0)
    )

    fp = np.sum(
        (predictions == 1)
        & (targets == 0)
    )

    fn = np.sum(
        (predictions == 0)
        & (targets == 1)
    )

    accuracy = (
        (tp + tn)
        / max(len(targets), 1)
    )

    precision = (
        tp
        / max(tp + fp, 1)
    )

    recall = (
        tp
        / max(tp + fn, 1)
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
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []
    targets = []

    for X, y in train_loader:

        X = X.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X.size(0)
        )

        pred = torch.argmax(
            logits,
            dim=1,
        )

        predictions.extend(
            pred.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            y.detach()
            .cpu()
            .numpy()
        )

    metrics = calculate_metrics(
        predictions,
        targets,
    )

    metrics["loss"] = (
        running_loss
        / len(train_dataset)
    )

    return metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    running_loss = 0.0

    predictions = []
    targets = []

    for X, y in loader:

        X = X.to(DEVICE)

        y = y.to(DEVICE)

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        running_loss += (
            loss.item()
            * X.size(0)
        )

        pred = torch.argmax(
            logits,
            dim=1,
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        targets.extend(
            y.cpu().numpy()
        )

    metrics = calculate_metrics(
        predictions,
        targets,
    )

    metrics["loss"] = (
        running_loss
        / len(loader.dataset)
    )

    return metrics


# ============================================================
# TRAIN
# ============================================================

best_val_f1 = -1.0
best_epoch = 0

history = []

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    train_metrics = train_one_epoch()

    val_metrics = evaluate(
        val_loader
    )

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train | "
        f"Loss {train_metrics['loss']:.4f} | "
        f"Acc {train_metrics['accuracy']:.4f} | "
        f"F1 {train_metrics['f1']:.4f}"
    )

    print(
        f"Val   | "
        f"Loss {val_metrics['loss']:.4f} | "
        f"Acc {val_metrics['accuracy']:.4f} | "
        f"Precision {val_metrics['precision']:.4f} | "
        f"Recall {val_metrics['recall']:.4f} | "
        f"F1 {val_metrics['f1']:.4f}"
    )

    history.append(
        {
            "epoch": epoch,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_f1":
                train_metrics["f1"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],
        }
    )

    # Save best model using validation F1
    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]

        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_channels": 4,

                "input_length":
                    X_train.shape[-1],

                "classes": {
                    0: "no_human",
                    1: "human",
                },

                "epoch": epoch,

                "val_metrics":
                    val_metrics,
            },
            MODEL_DIR
            / "radar_human_detector_v2.pt",
        )

        print(
            f"  -> Best model saved "
            f"(Val F1 = {best_val_f1:.4f})"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    MODEL_DIR
    / "radar_training_history_v2.csv"
)

with open(
    history_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=history[0].keys(),
    )

    writer.writeheader()

    writer.writerows(history)


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_DIR
    / "radar_human_detector_v2.pt",
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

test_metrics = evaluate(
    test_loader
)

print(
    f"Test accuracy:  {test_metrics['accuracy']:.4f}"
)

print(
    f"Test precision: {test_metrics['precision']:.4f}"
)

print(
    f"Test recall:    {test_metrics['recall']:.4f}"
)

print(
    f"Test F1:        {test_metrics['f1']:.4f}"
)

print("\nConfusion matrix:")

print(
    f"  True Negative:  {test_metrics['tn']}"
)

print(
    f"  False Positive: {test_metrics['fp']}"
)

print(
    f"  False Negative: {test_metrics['fn']}"
)

print(
    f"  True Positive:  {test_metrics['tp']}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RADAR TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test F1: "
    f"{test_metrics['f1']:.4f}"
)

print()
print(
    f"Model saved to:\n"
    f"{MODEL_DIR / 'radar_human_detector_v2.pt'}"
)

print(
    f"\nHistory saved to:\n"
    f"{history_path}"
)

print("=" * 70)