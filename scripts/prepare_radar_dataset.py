from pathlib import Path
import csv
import ast
import json
import random
import gc

import numpy as np


BASE = Path(__file__).resolve().parents[1]

RADAR_DIR = BASE / "data" / "raw" / "radar"
OUT_DIR = BASE / "data" / "processed" / "radar"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N0 = RADAR_DIR / "N0.csv"
N1 = RADAR_DIR / "N1.csv"

SEED = 42

# Keep this manageable for the first real model.
# We can scale later.
MAX_PER_CLASS = 10000

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(SEED)
np.random.seed(SEED)


def parse_complex_array(text):
    """
    Convert the string representation:

    [(a+bj), (c+dj), ...]

    into a numpy complex64 array.
    """

    try:
        values = ast.literal_eval(text)

        arr = np.asarray(values, dtype=np.complex64)

        return arr

    except Exception as e:
        raise ValueError(
            f"Could not parse Complex_data: {e}"
        )


def extract_features(complex_data):
    """
    Convert complex radar signal into a compact
    real-valued representation.

    Channels:
      0 = magnitude
      1 = real
      2 = imaginary
      3 = phase
    """

    z = parse_complex_array(complex_data)

    magnitude = np.abs(z).astype(np.float32)

    real = z.real.astype(np.float32)

    imag = z.imag.astype(np.float32)

    phase = np.angle(z).astype(np.float32)

    # Per-sample normalization of magnitude
    mag_mean = magnitude.mean()
    mag_std = magnitude.std()

    if mag_std > 1e-8:
        magnitude = (magnitude - mag_mean) / mag_std
    else:
        magnitude = magnitude - mag_mean

    features = np.stack(
        [
            magnitude,
            real,
            imag,
            phase,
        ],
        axis=0,
    )

    return features.astype(np.float32)


def read_dataset(
    path,
    target_label,
    max_samples,
):

    print("\n" + "=" * 70)
    print(f"READING {path.name}")
    print("=" * 70)

    print(f"Target label: {target_label}")
    print(f"Maximum samples: {max_samples:,}")

    samples = []
    distances = []

    skipped = 0
    total = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            total += 1

            if len(samples) >= max_samples:
                break

            try:

                # ------------------------------------------------
                # Verify label
                # ------------------------------------------------

                presence = int(
                    float(row["Presence"])
                )

                if presence != target_label:
                    continue

                # ------------------------------------------------
                # Radar features
                # ------------------------------------------------

                features = extract_features(
                    row["Complex_data"]
                )

                # ------------------------------------------------
                # Distance
                # ------------------------------------------------

                try:
                    distance = float(
                        row["Distance"]
                    )
                except Exception:
                    distance = np.nan

                samples.append(features)
                distances.append(distance)

            except Exception as e:

                skipped += 1

                if skipped <= 5:
                    print(
                        f"Skipping row {total}: {e}"
                    )

            if len(samples) % 1000 == 0 and len(samples) > 0:

                print(
                    f"  collected "
                    f"{len(samples):,} samples...",
                    end="\r",
                )

    print()

    print(f"Rows scanned: {total:,}")
    print(f"Samples collected: {len(samples):,}")
    print(f"Rows skipped: {skipped:,}")

    if not samples:
        raise RuntimeError(
            f"No valid samples found in {path}"
        )

    X = np.stack(samples).astype(np.float32)

    y = np.full(
        len(samples),
        target_label,
        dtype=np.int64,
    )

    d = np.asarray(
        distances,
        dtype=np.float32,
    )

    print(f"Feature shape: {X.shape}")

    return X, y, d


def stratified_split(
    X,
    y,
    distances,
):

    indices = np.arange(len(X))

    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
    distances = distances[indices]

    n = len(X)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = train_end + int(
        n * VAL_RATIO
    )

    return (
        X[:train_end],
        y[:train_end],
        distances[:train_end],

        X[train_end:val_end],
        y[train_end:val_end],
        distances[train_end:val_end],

        X[val_end:],
        y[val_end:],
        distances[val_end:],
    )


def save_dataset(
    X_train,
    y_train,
    d_train,

    X_val,
    y_val,
    d_val,

    X_test,
    y_test,
    d_test,
):

    print("\nSaving datasets...")

    np.save(
        OUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUT_DIR / "distance_train.npy",
        d_train,
    )

    np.save(
        OUT_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        OUT_DIR / "y_val.npy",
        y_val,
    )

    np.save(
        OUT_DIR / "distance_val.npy",
        d_val,
    )

    np.save(
        OUT_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        OUT_DIR / "y_test.npy",
        y_test,
    )

    np.save(
        OUT_DIR / "distance_test.npy",
        d_test,
    )


def main():

    print("=" * 70)
    print("SENSE-X RADAR DATASET PREPARATION")
    print("=" * 70)

    # ========================================================
    # LOAD NO-HUMAN
    # ========================================================

    X0, y0, d0 = read_dataset(
        N0,
        target_label=0,
        max_samples=MAX_PER_CLASS,
    )

    # ========================================================
    # LOAD HUMAN
    # ========================================================

    X1, y1, d1 = read_dataset(
        N1,
        target_label=1,
        max_samples=MAX_PER_CLASS,
    )

    # ========================================================
    # COMBINE
    # ========================================================

    print("\n" + "=" * 70)
    print("COMBINING")
    print("=" * 70)

    X = np.concatenate(
        [X0, X1],
        axis=0,
    )

    y = np.concatenate(
        [y0, y1],
        axis=0,
    )

    distances = np.concatenate(
        [d0, d1],
        axis=0,
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print(
        f"No-human samples: "
        f"{np.sum(y == 0):,}"
    )

    print(
        f"Human samples: "
        f"{np.sum(y == 1):,}"
    )

    # ========================================================
    # SHUFFLE
    # ========================================================

    indices = np.arange(len(X))

    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
    distances = distances[indices]

    # ========================================================
    # SPLIT
    # ========================================================

    (
        X_train,
        y_train,
        d_train,

        X_val,
        y_val,
        d_val,

        X_test,
        y_test,
        d_test,

    ) = stratified_split(
        X,
        y,
        distances,
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_dataset(
        X_train,
        y_train,
        d_train,

        X_val,
        y_val,
        d_val,

        X_test,
        y_test,
        d_test,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "feature_channels": [
            "normalized_magnitude",
            "real",
            "imaginary",
            "phase",
        ],
        "num_range_bins": int(X.shape[-1]),
        "classes": {
            "0": "no_human",
            "1": "human",
        },
        "max_per_class": MAX_PER_CLASS,
        "seed": SEED,
    }

    with open(
        OUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("RADAR DATASET READY")
    print("=" * 70)

    print(
        f"Train: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation: "
        f"{len(X_val):,}"
    )

    print(
        f"Test: "
        f"{len(X_test):,}"
    )

    print(
        f"Feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"\nSaved to:"
    )

    print(OUT_DIR)

    print("=" * 70)


if __name__ == "__main__":
    main()