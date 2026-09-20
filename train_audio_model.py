# ================================================================
# SENSE-X AUDIO MODEL V2
# DroneAudioSet: Drone-only vs Source-present
# Recording-level train/validation split
# ================================================================

import os
import random
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, get_dataset_split_names

import librosa


# ================================================================
# CONFIGURATION
# ================================================================

DATASET = "ahlab-drone-project/DroneAudioSet"

OUTPUT_DIR = "data/processed/audio"
MODEL_DIR = "models/audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Audio
SAMPLE_RATE = 16000

CLIP_SECONDS = 3
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# Dataset selection
# We intentionally keep this small enough for a fast hackathon experiment.
MAX_SPLITS = 5
MAX_RECORDINGS_PER_CLASS = 20
MAX_CLIPS_PER_RECORDING = 8

# Training
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.80

SEED = 42

DEVICE = torch.device("cpu")


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("SENSE-X AUDIO MODEL V2")
print("=" * 70)

print("\nConfiguration:")
print(f"Sample rate       : {SAMPLE_RATE}")
print(f"Clip duration     : {CLIP_SECONDS}s")
print(f"Mel bands         : {N_MELS}")
print(f"Max splits        : {MAX_SPLITS}")
print(f"Max recordings/class: {MAX_RECORDINGS_PER_CLASS}")
print(f"Clips/recording   : {MAX_CLIPS_PER_RECORDING}")
print(f"Epochs            : {EPOCHS}")
print(f"Device            : {DEVICE}")


# ================================================================
# DATASET SPLITS
# ================================================================

print("\n" + "=" * 70)
print("DISCOVERING DATASET SPLITS")
print("=" * 70)

configs = {
    "drone_only": "drone-only",
    "source_present": "drone-with-source",
}

split_candidates = []

try:
    splits = get_dataset_split_names(
        DATASET,
        config_name="drone-only"
    )

    print("\nAvailable drone-only splits:")
    print(splits[:20])

    # Only use train_XXX splits.
    train_splits = [
        s for s in splits
        if s.startswith("train_")
    ]

    split_candidates = train_splits[:MAX_SPLITS]

except Exception as e:
    print("Could not discover splits automatically.")
    print("Error:", e)

    split_candidates = [
        "train_001",
        "train_002",
        "train_003",
        "train_004",
        "train_005",
    ]


print("\nSplits selected:")

for split in split_candidates:
    print(" -", split)


# ================================================================
# COLLECT RECORDING METADATA
# ================================================================

def collect_recordings(config_name, class_label, max_recordings):
    """
    Collect recording metadata from several dataset splits.

    class_label:
        0 = drone_only
        1 = source_present
    """

    print("\n" + "-" * 70)

    if class_label == 0:
        print("Collecting DRONE-ONLY recordings...")
    else:
        print("Collecting SOURCE-PRESENT recordings...")

    selected = []

    for split in split_candidates:

        if len(selected) >= max_recordings:
            break

        print(f"\nReading {config_name} / {split}")

        try:

            dataset = load_dataset(
                DATASET,
                config_name,
                split=split,
                streaming=True
            )

            for sample in dataset:

                file_path = sample["file_path"]

                # Avoid duplicate recording paths.
                if any(
                    item["file_path"] == file_path
                    for item in selected
                ):
                    continue

                selected.append({
                    "config": config_name,
                    "split": split,
                    "file_path": file_path,
                    "label": class_label,
                })

                print(
                    f"  selected {len(selected):02d}: "
                    f"{os.path.basename(file_path)}"
                )

                if len(selected) >= max_recordings:
                    break

        except Exception as e:

            print(
                f"  WARNING: Could not read {split}: {e}"
            )

    return selected


# ================================================================
# COLLECT BOTH CLASSES
# ================================================================

drone_records = collect_recordings(
    "drone-only",
    0,
    MAX_RECORDINGS_PER_CLASS
)

source_records = collect_recordings(
    "drone-with-source",
    1,
    MAX_RECORDINGS_PER_CLASS
)


print("\n" + "=" * 70)
print("RECORDINGS SELECTED")
print("=" * 70)

print("Drone-only    :", len(drone_records))
print("Source-present:", len(source_records))


if len(drone_records) < 5 or len(source_records) < 5:

    raise RuntimeError(
        "\nNot enough recordings collected.\n"
        "At least 5 recordings per class are required."
    )


# ================================================================
# RECORDING-LEVEL TRAIN / VALIDATION SPLIT
# ================================================================

def split_recordings(records):

    records = records.copy()

    random.shuffle(records)

    split_index = int(
        len(records) * TRAIN_RATIO
    )

    # Make sure validation is never empty.
    split_index = max(
        1,
        min(split_index, len(records) - 1)
    )

    train = records[:split_index]
    val = records[split_index:]

    return train, val


drone_train, drone_val = split_recordings(
    drone_records
)

source_train, source_val = split_recordings(
    source_records
)


train_records = (
    drone_train +
    source_train
)

val_records = (
    drone_val +
    source_val
)

random.shuffle(train_records)
random.shuffle(val_records)


print("\n" + "=" * 70)
print("RECORDING-LEVEL SPLIT")
print("=" * 70)

print(
    f"Train recordings: {len(train_records)}"
)

print(
    f"Validation recordings: {len(val_records)}"
)

print(
    f"Train drone-only: "
    f"{sum(r['label'] == 0 for r in train_records)}"
)

print(
    f"Train source-present: "
    f"{sum(r['label'] == 1 for r in train_records)}"
)

print(
    f"Val drone-only: "
    f"{sum(r['label'] == 0 for r in val_records)}"
)

print(
    f"Val source-present: "
    f"{sum(r['label'] == 1 for r in val_records)}"
)


# ================================================================
# SAVE RECORDING MANIFEST
# ================================================================

manifest_rows = []

for record in train_records:

    manifest_rows.append({
        "file_path": record["file_path"],
        "config": record["config"],
        "split": record["split"],
        "label": record["label"],
        "dataset_split": "train",
    })


for record in val_records:

    manifest_rows.append({
        "file_path": record["file_path"],
        "config": record["config"],
        "split": record["split"],
        "label": record["label"],
        "dataset_split": "validation",
    })


manifest_df = pd.DataFrame(manifest_rows)

manifest_path = os.path.join(
    OUTPUT_DIR,
    "audio_manifest_v2.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False
)

print(
    "\nManifest saved:"
)

print(manifest_path)


# ================================================================
# AUDIO LOADING
# ================================================================

def load_recording(record):

    dataset = load_dataset(
        DATASET,
        record["config"],
        split=record["split"],
        streaming=True
    )

    target_path = record["file_path"]

    for sample in dataset:

        if sample["file_path"] == target_path:

            audio = sample["audio"]

            y = np.asarray(
                audio["array"],
                dtype=np.float32
            )

            y = np.squeeze(y)

            sr = int(
                audio["sampling_rate"]
            )

            return y, sr

    return None, None


# ================================================================
# LOG-MEL FEATURE EXTRACTION
# ================================================================

def extract_features(y, sr):

    # Convert stereo/multichannel to mono.
    if y.ndim > 1:

        y = np.mean(
            y,
            axis=-1
        )

    y = y.astype(
        np.float32
    )

    # Resample if required.
    if sr != SAMPLE_RATE:

        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

        sr = SAMPLE_RATE

    if len(y) < CLIP_SAMPLES:

        return []

    clips = []

    # Select evenly spaced clips.
    possible_starts = (
        len(y) - CLIP_SAMPLES
    )

    if possible_starts <= 0:

        starts = [0]

    else:

        max_clips = min(
            MAX_CLIPS_PER_RECORDING,
            possible_starts // CLIP_SAMPLES + 1
        )

        starts = np.linspace(
            0,
            possible_starts,
            num=max_clips,
            dtype=int
        )

    for start in starts:

        clip = y[
            start:start + CLIP_SAMPLES
        ]

        if len(clip) < CLIP_SAMPLES:
            continue

        # Remove DC offset.
        clip = clip - np.mean(clip)

        # Normalize amplitude.
        peak = np.max(
            np.abs(clip)
        )

        if peak > 1e-8:

            clip = clip / peak

        # Mel spectrogram.
        mel = librosa.feature.melspectrogram(
            y=clip,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=20,
            fmax=8000
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Per-example standardization.
        mean = np.mean(mel_db)
        std = np.std(mel_db)

        if std > 1e-6:

            mel_db = (
                (mel_db - mean)
                / std
            )

        clips.append(
            mel_db.astype(
                np.float32
            )
        )

    return clips


# ================================================================
# BUILD FEATURE DATASET
# ================================================================

def build_feature_dataset(records, dataset_name):

    X = []
    y = []
    groups = []

    print("\n" + "=" * 70)
    print(f"BUILDING {dataset_name.upper()} FEATURES")
    print("=" * 70)

    for index, record in enumerate(records):

        filename = os.path.basename(
            record["file_path"]
        )

        print(
            f"[{index + 1}/{len(records)}] "
            f"{filename} | "
            f"label={record['label']} | "
            f"split={record['split']}"
        )

        try:

            waveform, sr = load_recording(
                record
            )

            if waveform is None:

                print("  WARNING: audio not found")
                continue

            features = extract_features(
                waveform,
                sr
            )

            print(
                f"  clips extracted: "
                f"{len(features)}"
            )

            for feature in features:

                X.append(feature)
                y.append(record["label"])

                # Group ID is the recording itself.
                groups.append(
                    record["file_path"]
                )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    if not X:

        return (
            np.empty(
                (0, 1, N_MELS, 188),
                dtype=np.float32
            ),
            np.array(
                [],
                dtype=np.int64
            ),
            []
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    # Add CNN channel dimension.
    X = X[:, np.newaxis, :, :]

    y = np.asarray(
        y,
        dtype=np.int64
    )

    return X, y, groups


# ================================================================
# TRAIN FEATURES
# ================================================================

X_train, y_train, train_groups = (
    build_feature_dataset(
        train_records,
        "training"
    )
)


# ================================================================
# VALIDATION FEATURES
# ================================================================

X_val, y_val, val_groups = (
    build_feature_dataset(
        val_records,
        "validation"
    )
)


print("\n" + "=" * 70)
print("FEATURE DATASET")
print("=" * 70)

print(
    "Training features:",
    X_train.shape
)

print(
    "Validation features:",
    X_val.shape
)

print(
    "Training labels:",
    np.bincount(y_train)
)

print(
    "Validation labels:",
    np.bincount(y_val)
)


if len(X_train) == 0 or len(X_val) == 0:

    raise RuntimeError(
        "Feature extraction produced an empty dataset."
    )


# ================================================================
# PYTORCH DATASET
# ================================================================

class AudioDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


train_dataset = AudioDataset(
    X_train,
    y_train
)

val_dataset = AudioDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# CNN MODEL
# ================================================================

class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


model = AudioCNN().to(
    DEVICE
)


# ================================================================
# LOSS / OPTIMIZER
# ================================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ================================================================
# TRAINING
# ================================================================

print("\n" + "=" * 70)
print("STARTING V2 TRAINING")
print("=" * 70)


best_val_accuracy = 0.0

history = []


for epoch in range(EPOCHS):

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            batch_X
        )

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch_X.size(0)
        )

        predictions = (
            torch.argmax(
                outputs,
                dim=1
            )
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_loss = (
        total_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            outputs = model(
                batch_X
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += (
                batch_y.size(0)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.cpu().numpy()
            )

    val_accuracy = (
        val_correct / val_total
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.3f} | "
        f"Val Acc: {val_accuracy:.3f}"
    )


    history.append({
        "epoch": epoch + 1,
        "loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy
    })


    # ------------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        model_path = os.path.join(
            MODEL_DIR,
            "audio_source_detector_v2.pt"
        )

        torch.save(
            model.state_dict(),
            model_path
        )


# ================================================================
# CONFUSION MATRIX
# ================================================================

confusion = np.zeros(
    (2, 2),
    dtype=int
)

for target, prediction in zip(
    all_targets,
    all_predictions
):

    confusion[
        target,
        prediction
    ] += 1


print("\n" + "=" * 70)
print("VALIDATION CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                Predicted"
)

print(
    "              Drone  Source"
)

print(
    f"Actual Drone   "
    f"{confusion[0,0]:5d} "
    f"{confusion[0,1]:6d}"
)

print(
    f"Actual Source  "
    f"{confusion[1,0]:5d} "
    f"{confusion[1,1]:6d}"
)


# ================================================================
# SAVE TRAINING HISTORY
# ================================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "audio_training_history_v2.csv"
)

pd.DataFrame(
    history
).to_csv(
    history_path,
    index=False
)


# ================================================================
# SAVE MODEL CONFIGURATION
# ================================================================

config = {

    "model": "AudioCNN",

    "dataset":
        "ahlab-drone-project/DroneAudioSet",

    "classes": {
        "0": "drone_only",
        "1": "source_present"
    },

    "sample_rate":
        SAMPLE_RATE,

    "clip_seconds":
        CLIP_SECONDS,

    "n_mels":
        N_MELS,

    "n_fft":
        N_FFT,

    "hop_length":
        HOP_LENGTH,

    "train_recordings":
        len(train_records),

    "validation_recordings":
        len(val_records),

    "train_examples":
        len(X_train),

    "validation_examples":
        len(X_val),

    "best_validation_accuracy":
        best_val_accuracy,

    "seed":
        SEED
}


config_path = os.path.join(
    MODEL_DIR,
    "audio_source_detector_v2_config.json"
)

with open(
    config_path,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy:.3f}"
)

print(
    "\nModel saved:"
)

print(
    os.path.join(
        MODEL_DIR,
        "audio_source_detector_v2.pt"
    )
)

print(
    "\nConfig saved:"
)

print(config_path)

print(
    "\nTraining history:"
)

print(history_path)

print("\nClasses:")

print(
    "0 = drone_only"
)

print(
    "1 = source_present"
)

print("\nNext:")
print(
    "Connect the source-present probability "
    "to the SENSE-X sensor-fusion engine."
)

print("=" * 70)