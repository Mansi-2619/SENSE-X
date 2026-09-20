from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "ahlab-drone-project/DroneAudioSet"

SAMPLE_RATE = 16000
CLIP_SECONDS = 3
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MAX_SPLITS = 5
MAX_RECORDINGS_PER_CLASS = 20
CLIPS_PER_RECORDING = 8

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

DEVICE = "cpu"

SEED = 42


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "models/audio/audio_source_detector_v2.pt"
)

CONFIG_PATH = Path(
    "models/audio/audio_source_detector_v2_config.json"
)

MANIFEST_PATH = Path(
    "data/processed/audio/audio_manifest_v2.csv"
)

HISTORY_PATH = Path(
    "data/processed/audio/audio_training_history_v2.csv"
)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("SENSE-X AUDIO MODEL V2")
print("=" * 70)

print("\nConfiguration:")
print("Dataset:", DATASET)
print("Sample rate:", SAMPLE_RATE)
print("Clip duration:", CLIP_SECONDS, "seconds")
print("Mel bands:", N_MELS)
print("Max splits:", MAX_SPLITS)
print("Max recordings/class:", MAX_RECORDINGS_PER_CLASS)
print("Clips/recording:", CLIPS_PER_RECORDING)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Device:", DEVICE)
print("Model path:", MODEL_PATH)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def get_recordings(config, max_recordings):
    """
    Collect recording paths from the requested DroneAudioSet config.
    """

    print()
    print("=" * 70)
    print("Collecting:", config)
    print("=" * 70)

    # Get available splits
    from datasets import get_dataset_split_names

    splits = get_dataset_split_names(
        DATASET,
        config_name=config
    )

    print("Available splits:", splits[:MAX_SPLITS])

    recordings = []

    for split in splits[:MAX_SPLITS]:

        print(
            f"\nInspecting split: {split}"
        )

        try:

            dataset = load_dataset(
                DATASET,
                config,
                split=split,
                streaming=True
            )

            for sample in dataset:

                file_path = sample["file_path"]

                recordings.append({
                    "config": config,
                    "split": split,
                    "file_path": file_path
                })

                if len(recordings) >= max_recordings:
                    break

        except Exception as e:

            print(
                f"ERROR in {split}: {e}"
            )

        if len(recordings) >= max_recordings:
            break

    print(
        f"Collected {len(recordings)} recordings"
    )

    return recordings


# ============================================================
# COLLECT DATA
# ============================================================

drone_recordings = get_recordings(
    "drone-only",
    MAX_RECORDINGS_PER_CLASS
)

source_recordings = get_recordings(
    "drone-with-source",
    MAX_RECORDINGS_PER_CLASS
)


# ============================================================
# MANIFEST
# ============================================================

manifest_rows = []

for item in drone_recordings:

    manifest_rows.append({
        "config": item["config"],
        "split": item["split"],
        "file_path": item["file_path"],
        "label": 0
    })


for item in source_recordings:

    manifest_rows.append({
        "config": item["config"],
        "split": item["split"],
        "file_path": item["file_path"],
        "label": 1
    })


manifest = pd.DataFrame(manifest_rows)

manifest.to_csv(
    MANIFEST_PATH,
    index=False
)

print("\nManifest saved:")
print(MANIFEST_PATH)

print("\nClass distribution:")

print(
    manifest["label"].value_counts().sort_index()
)


# ============================================================
# RECORDING-LEVEL TRAIN/VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("RECORDING-LEVEL SPLIT")
print("=" * 70)

train_rows = []
val_rows = []

for label in [0, 1]:

    class_rows = manifest[
        manifest["label"] == label
    ].sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    split_index = int(
        len(class_rows) * 0.80
    )

    train_rows.append(
        class_rows.iloc[:split_index]
    )

    val_rows.append(
        class_rows.iloc[split_index:]
    )


train_manifest = pd.concat(
    train_rows,
    ignore_index=True
)

val_manifest = pd.concat(
    val_rows,
    ignore_index=True
)

print(
    "Training recordings:",
    len(train_manifest)
)

print(
    "Validation recordings:",
    len(val_manifest)
)

print(
    "\nTraining class counts:"
)

print(
    train_manifest["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nValidation class counts:"
)

print(
    val_manifest["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# LOAD A RECORDING FROM HUGGING FACE
# ============================================================

def load_recording(row):

    config = row["config"]
    split = row["split"]
    target_path = row["file_path"]

    dataset = load_dataset(
        DATASET,
        config,
        split=split,
        streaming=True
    )

    for sample in dataset:

        if sample["file_path"] == target_path:

            audio = sample["audio"]

            y = np.asarray(
                audio["array"],
                dtype=np.float32
            )

            sr = int(
                audio["sampling_rate"]
            )

            return y, sr

    raise RuntimeError(
        f"Could not find recording: {target_path}"
    )


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def prepare_audio(y, sr):

    # Convert multichannel → mono
    if y.ndim > 1:

        y = np.mean(
            y,
            axis=-1
        )

    y = y.astype(
        np.float32
    )

    # Resample if necessary
    if sr != SAMPLE_RATE:

        y = librosa.resample(
            y=y,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    # Remove DC offset
    y = y - np.mean(y)

    return y


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mel(y):

    if len(y) < CLIP_SAMPLES:

        y = np.pad(
            y,
            (
                0,
                CLIP_SAMPLES - len(y)
            )
        )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize each example
    mean = mel_db.mean()
    std = mel_db.std()

    if std > 1e-6:

        mel_db = (
            mel_db - mean
        ) / std

    else:

        mel_db = (
            mel_db - mean
        )

    return mel_db.astype(
        np.float32
    )


# ============================================================
# EXTRACT CLIPS
# ============================================================

def extract_clips_from_recording(
    y,
    number_of_clips
):

    if len(y) < CLIP_SAMPLES:

        return [
            np.pad(
                y,
                (
                    0,
                    CLIP_SAMPLES - len(y)
                )
            )
        ]

    max_start = (
        len(y) - CLIP_SAMPLES
    )

    if max_start == 0:

        starts = [0]

    else:

        starts = np.linspace(
            0,
            max_start,
            number_of_clips
        ).astype(int)

    clips = []

    for start in starts:

        clip = y[
            start:
            start + CLIP_SAMPLES
        ]

        if len(clip) < CLIP_SAMPLES:

            clip = np.pad(
                clip,
                (
                    0,
                    CLIP_SAMPLES - len(clip)
                )
            )

        clips.append(clip)

    return clips


# ============================================================
# BUILD FEATURE ARRAYS
# ============================================================

def build_features(
    recording_manifest,
    name
):

    print("\n" + "=" * 70)
    print("BUILDING FEATURES:", name)
    print("=" * 70)

    features = []
    labels = []

    for index, row in recording_manifest.iterrows():

        print(
            f"[{index + 1}/{len(recording_manifest)}] "
            f"{row['file_path']}"
        )

        try:

            y, sr = load_recording(
                row
            )

            y = prepare_audio(
                y,
                sr
            )

            clips = extract_clips_from_recording(
                y,
                CLIPS_PER_RECORDING
            )

            for clip in clips:

                mel = extract_mel(
                    clip
                )

                features.append(
                    mel
                )

                labels.append(
                    int(row["label"])
                )

        except Exception as e:

            print(
                "ERROR:",
                e
            )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    # CNN input shape:
    # [N, 1, Mel, Time]

    X = X[:, None, :, :]

    print(
        "\nFeature shape:",
        X.shape
    )

    print(
        "Labels:",
        np.bincount(y)
    )

    return X, y


X_train, y_train = build_features(
    train_manifest,
    "TRAIN"
)

X_val, y_val = build_features(
    val_manifest,
    "VALIDATION"
)


# ============================================================
# PYTORCH DATASET
# ============================================================

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

    def __getitem__(
        self,
        index
    ):

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


# ============================================================
# CNN MODEL
# ============================================================

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

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = AudioCNN().to(
    DEVICE
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


best_val_acc = -1.0
best_epoch = 0

history = []


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X_batch.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == y_batch
        ).sum().item()

        train_total += (
            y_batch.size(0)
        )

    train_loss = (
        running_loss /
        train_total
    )

    train_acc = (
        train_correct /
        train_total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            outputs = model(
                X_batch
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == y_batch
            ).sum().item()

            val_total += (
                y_batch.size(0)
            )

    val_acc = (
        val_correct /
        val_total
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.3f} | "
        f"Val Acc: {val_acc:.3f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_epoch = (
            epoch + 1
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  -> BEST MODEL SAVED "
            f"(epoch {best_epoch}, "
            f"val_acc={best_val_acc:.3f})"
        )

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_acc,

        "validation_accuracy": val_acc,

        "best_validation_accuracy":
            best_val_acc

    })


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# SAVE MODEL CONFIG
# ============================================================

model_config = {

    "sample_rate": SAMPLE_RATE,

    "clip_seconds": CLIP_SECONDS,

    "n_mels": N_MELS,

    "n_fft": N_FFT,

    "hop_length": HOP_LENGTH,

    "classes": {

        "0": "drone_only",

        "1": "source_present"

    },

    "best_epoch": best_epoch,

    "best_validation_accuracy":
        best_val_acc

}


with open(
    CONFIG_PATH,
    "w"
) as f:

    json.dump(
        model_config,
        f,
        indent=4
    )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("RESTORING BEST MODEL")
print("=" * 70)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation accuracy:",
    f"{best_val_acc:.3f}"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print(
    "Best checkpoint restored:"
)

print(
    MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        outputs = model(
            X_batch
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )


all_predictions = np.asarray(
    all_predictions
)

all_labels = np.asarray(
    all_labels
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (2, 2),
    dtype=int
)

for actual, predicted in zip(
    all_labels,
    all_predictions
):

    confusion[
        actual,
        predicted
    ] += 1


print("\n" + "=" * 70)
print("FINAL CONFUSION MATRIX")
print("=" * 70)

print(
    "              Predicted"
)

print(
    "              Drone  Source"
)

print(
    f"Actual Drone  "
    f"{confusion[0,0]:5d} "
    f"{confusion[0,1]:6d}"
)

print(
    f"Actual Source  "
    f"{confusion[1,0]:5d} "
    f"{confusion[1,1]:6d}"
)


# ============================================================
# FINAL ACCURACY
# ============================================================

final_accuracy = np.mean(
    all_predictions ==
    all_labels
)

print(
    "\nFinal accuracy using BEST checkpoint:",
    f"{final_accuracy:.3f}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("AUDIO MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Config:",
    CONFIG_PATH
)

print(
    "History:",
    HISTORY_PATH
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation accuracy:",
    f"{best_val_acc:.3f}"
)

print(
    "Final accuracy:",
    f"{final_accuracy:.3f}"
)

print(
    "\nClasses:"
)

print(
    "0 = drone_only"
)

print(
    "1 = source_present"
)