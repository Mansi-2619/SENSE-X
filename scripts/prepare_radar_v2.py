from pathlib import Path
import csv
import ast
import json
import random
from collections import defaultdict

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[1]

RAW_DIR = (
    BASE
    / "data"
    / "raw"
    / "radar"
)

OUT_DIR = (
    BASE
    / "data"
    / "processed"
    / "radar_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N0_PATH = RAW_DIR / "N0.csv"
N1_PATH = RAW_DIR / "N1.csv"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# We deliberately limit samples PER EXPERIMENT, not per class.
#
# 22 N0 IDs × 250 ~= 5,500
# 88 N1 IDs × 250 ~= 22,000
#
# We then balance the classes after grouped splitting.
# ------------------------------------------------------------

MAX_SAMPLES_PER_ID = 250

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# COMPLEX RADAR PROCESSING
# ============================================================

def parse_complex_array(text):
    """
    Parse one Complex_data CSV field into complex64 values.
    """

    values = ast.literal_eval(text)

    return np.asarray(
        values,
        dtype=np.complex64,
    )


def extract_features(complex_data):
    """
    Produce the same 4 x 180 representation used by the
    original SENSE-X radar pipeline.

    Channels:
        0 magnitude normalized per sample
        1 real
        2 imaginary
        3 phase
    """

    z = parse_complex_array(
        complex_data
    )

    magnitude = np.abs(
        z
    ).astype(
        np.float32
    )

    real = z.real.astype(
        np.float32
    )

    imaginary = z.imag.astype(
        np.float32
    )

    phase = np.angle(
        z
    ).astype(
        np.float32
    )

    mean = float(
        magnitude.mean()
    )

    std = float(
        magnitude.std()
    )

    if std > 1e-8:
        magnitude = (
            magnitude - mean
        ) / std
    else:
        magnitude = (
            magnitude - mean
        )

    features = np.stack(
        [
            magnitude,
            real,
            imaginary,
            phase,
        ],
        axis=0,
    )

    return features.astype(
        np.float32
    )


# ============================================================
# DISCOVER EXPERIMENT IDS
# ============================================================

def discover_ids(
    path,
    expected_label,
):
    """
    Read only metadata fields and determine which ID_code
    groups exist in a source file.
    """

    print(
        f"\nDiscovering experiment IDs in "
        f"{path.name}..."
    )

    counts = defaultdict(int)

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            try:
                presence = int(
                    float(
                        row["Presence"]
                    )
                )
            except Exception:
                continue

            if presence != expected_label:
                continue

            experiment_id = (
                row["ID_code"].strip()
            )

            if not experiment_id:
                continue

            counts[
                experiment_id
            ] += 1

    print(
        f"Found {len(counts)} "
        f"experiment IDs."
    )

    return dict(counts)


# ============================================================
# GROUP SPLIT
# ============================================================

def split_ids(
    ids,
    seed,
):
    """
    Split complete experiment IDs.

    No ID is allowed to occur in more than one split.
    """

    ids = list(ids)

    rng = random.Random(
        seed
    )

    rng.shuffle(
        ids
    )

    n = len(ids)

    n_train = max(
        1,
        int(
            n * TRAIN_RATIO
        ),
    )

    n_val = max(
        1,
        int(
            n * VAL_RATIO
        ),
    )

    # Guarantee at least one test group.
    if (
        n_train
        + n_val
        >= n
    ):
        n_train = max(
            1,
            n - 2
        )

        n_val = 1

    train_ids = set(
        ids[
            :n_train
        ]
    )

    val_ids = set(
        ids[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = set(
        ids[
            n_train + n_val:
        ]
    )

    if not test_ids:
        raise RuntimeError(
            "Grouped split produced "
            "no test IDs."
        )

    assert train_ids.isdisjoint(
        val_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert val_ids.isdisjoint(
        test_ids
    )

    return (
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# ASSIGN SPLIT
# ============================================================

def build_split_lookup(
    train_ids,
    val_ids,
    test_ids,
):
    lookup = {}

    for code in train_ids:
        lookup[code] = "train"

    for code in val_ids:
        lookup[code] = "val"

    for code in test_ids:
        lookup[code] = "test"

    return lookup


# ============================================================
# SAMPLE FROM EVERY EXPERIMENT
# ============================================================

def collect_samples(
    path,
    expected_label,
    split_lookup,
):
    """
    Collect at most MAX_SAMPLES_PER_ID samples from each
    experiment.

    Since each ID contains thousands of rows, uniformly spaced
    sampling is used instead of simply taking the first rows.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        f"COLLECTING {path.name}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # First pass: count valid rows for each ID
    # --------------------------------------------------------

    counts = defaultdict(int)

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            try:
                presence = int(
                    float(
                        row["Presence"]
                    )
                )
            except Exception:
                continue

            if presence != expected_label:
                continue

            experiment_id = (
                row["ID_code"].strip()
            )

            if experiment_id not in split_lookup:
                continue

            counts[
                experiment_id
            ] += 1

    # --------------------------------------------------------
    # Determine uniformly distributed row positions
    # --------------------------------------------------------

    wanted_positions = {}

    for experiment_id, count in counts.items():

        amount = min(
            MAX_SAMPLES_PER_ID,
            count,
        )

        positions = np.linspace(
            0,
            count - 1,
            num=amount,
            dtype=np.int64,
        )

        wanted_positions[
            experiment_id
        ] = set(
            int(x)
            for x in positions
        )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    data = {
        "train": [],
        "val": [],
        "test": [],
    }

    distances = {
        "train": [],
        "val": [],
        "test": [],
    }

    ids = {
        "train": [],
        "val": [],
        "test": [],
    }

    elapsed_times = {
        "train": [],
        "val": [],
        "test": [],
    }

    current_index = defaultdict(int)

    # --------------------------------------------------------
    # Second pass: extract selected rows
    # --------------------------------------------------------

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            try:
                presence = int(
                    float(
                        row["Presence"]
                    )
                )
            except Exception:
                continue

            if presence != expected_label:
                continue

            experiment_id = (
                row["ID_code"].strip()
            )

            if experiment_id not in split_lookup:
                continue

            local_index = (
                current_index[
                    experiment_id
                ]
            )

            current_index[
                experiment_id
            ] += 1

            if (
                local_index
                not in wanted_positions[
                    experiment_id
                ]
            ):
                continue

            try:
                features = extract_features(
                    row["Complex_data"]
                )

            except Exception as error:

                print(
                    f"Skipping "
                    f"{experiment_id} "
                    f"row {local_index}: "
                    f"{error}"
                )

                continue

            split = split_lookup[
                experiment_id
            ]

            try:
                distance = float(
                    row["Distance"]
                )

            except Exception:
                distance = np.nan

            try:
                elapsed = float(
                    row["Elapsed_Time"]
                )

            except Exception:
                elapsed = np.nan

            data[
                split
            ].append(
                features
            )

            distances[
                split
            ].append(
                distance
            )

            ids[
                split
            ].append(
                experiment_id
            )

            elapsed_times[
                split
            ].append(
                elapsed
            )

    return (
        data,
        distances,
        ids,
        elapsed_times,
    )


# ============================================================
# COMBINE CLASS DATA
# ============================================================

def combine_split(
    split,
    zero_data,
    zero_distances,
    zero_ids,
    zero_times,
    one_data,
    one_distances,
    one_ids,
    one_times,
):
    """
    Combine N0 and N1 data for one split and class-balance
    by downsampling the larger class.
    """

    X0 = zero_data[
        split
    ]

    X1 = one_data[
        split
    ]

    if not X0 or not X1:
        raise RuntimeError(
            f"{split}: one class "
            f"contains no samples."
        )

    rng = np.random.default_rng(
        SEED
        + {
            "train": 1,
            "val": 2,
            "test": 3,
        }[split]
    )

    target = min(
        len(X0),
        len(X1),
    )

    idx0 = rng.choice(
        len(X0),
        size=target,
        replace=False,
    )

    idx1 = rng.choice(
        len(X1),
        size=target,
        replace=False,
    )

    combined_X = []

    combined_y = []

    combined_d = []

    combined_ids = []

    combined_times = []

    for index in idx0:

        combined_X.append(
            X0[index]
        )

        combined_y.append(
            0
        )

        combined_d.append(
            zero_distances[
                split
            ][index]
        )

        combined_ids.append(
            zero_ids[
                split
            ][index]
        )

        combined_times.append(
            zero_times[
                split
            ][index]
        )

    for index in idx1:

        combined_X.append(
            X1[index]
        )

        combined_y.append(
            1
        )

        combined_d.append(
            one_distances[
                split
            ][index]
        )

        combined_ids.append(
            one_ids[
                split
            ][index]
        )

        combined_times.append(
            one_times[
                split
            ][index]
        )

    permutation = rng.permutation(
        len(combined_X)
    )

    X = np.stack(
        combined_X
    ).astype(
        np.float32
    )

    y = np.asarray(
        combined_y,
        dtype=np.int64,
    )

    d = np.asarray(
        combined_d,
        dtype=np.float32,
    )

    experiment_ids = np.asarray(
        combined_ids,
        dtype=str,
    )

    times = np.asarray(
        combined_times,
        dtype=np.float64,
    )

    return (
        X[permutation],
        y[permutation],
        d[permutation],
        experiment_ids[
            permutation
        ],
        times[
            permutation
        ],
    )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(
    name,
    X,
    y,
    distances,
    experiment_ids,
    elapsed_times,
):
    np.save(
        OUT_DIR
        / f"X_{name}.npy",
        X,
    )

    np.save(
        OUT_DIR
        / f"y_{name}.npy",
        y,
    )

    np.save(
        OUT_DIR
        / f"distance_{name}.npy",
        distances,
    )

    np.save(
        OUT_DIR
        / f"id_{name}.npy",
        experiment_ids,
    )

    np.save(
        OUT_DIR
        / f"elapsed_{name}.npy",
        elapsed_times,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_split_summary(
    name,
    X,
    y,
    ids,
):
    unique_ids = sorted(
        set(
            ids.tolist()
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        name.upper()
    )

    print(
        "=" * 70
    )

    print(
        "X:",
        X.shape,
    )

    print(
        "No human:",
        int(
            np.sum(
                y == 0
            )
        ),
    )

    print(
        "Human:",
        int(
            np.sum(
                y == 1
            )
        ),
    )

    print(
        "Experiment IDs:",
        len(
            unique_ids
        ),
    )

    for code in unique_ids:
        print(
            " ",
            code
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "SENSE-X RADAR V2"
    )

    print(
        "GROUP-SAFE THROUGH-WALL DATASET PREPARATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Discover all experiment groups
    # --------------------------------------------------------

    zero_counts = discover_ids(
        N0_PATH,
        0,
    )

    one_counts = discover_ids(
        N1_PATH,
        1,
    )

    # --------------------------------------------------------
    # Split IDs independently within each class
    # --------------------------------------------------------

    (
        zero_train,
        zero_val,
        zero_test,
    ) = split_ids(
        zero_counts.keys(),
        SEED,
    )

    (
        one_train,
        one_val,
        one_test,
    ) = split_ids(
        one_counts.keys(),
        SEED + 100,
    )

    # --------------------------------------------------------
    # Strong leakage assertions
    # --------------------------------------------------------

    all_train_ids = (
        zero_train
        | one_train
    )

    all_val_ids = (
        zero_val
        | one_val
    )

    all_test_ids = (
        zero_test
        | one_test
    )

    assert all_train_ids.isdisjoint(
        all_val_ids
    )

    assert all_train_ids.isdisjoint(
        all_test_ids
    )

    assert all_val_ids.isdisjoint(
        all_test_ids
    )

    print(
        "\nGrouped split:"
    )

    print(
        "TRAIN IDs:",
        len(
            all_train_ids
        ),
    )

    print(
        "VAL IDs:",
        len(
            all_val_ids
        ),
    )

    print(
        "TEST IDs:",
        len(
            all_test_ids
        ),
    )

    # --------------------------------------------------------
    # Build lookup tables
    # --------------------------------------------------------

    zero_lookup = build_split_lookup(
        zero_train,
        zero_val,
        zero_test,
    )

    one_lookup = build_split_lookup(
        one_train,
        one_val,
        one_test,
    )

    # --------------------------------------------------------
    # Extract selected samples
    # --------------------------------------------------------

    (
        zero_data,
        zero_distances,
        zero_ids,
        zero_times,
    ) = collect_samples(
        N0_PATH,
        0,
        zero_lookup,
    )

    (
        one_data,
        one_distances,
        one_ids,
        one_times,
    ) = collect_samples(
        N1_PATH,
        1,
        one_lookup,
    )

    # --------------------------------------------------------
    # Combine/balance each split
    # --------------------------------------------------------

    outputs = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        outputs[
            split
        ] = combine_split(
            split,

            zero_data,
            zero_distances,
            zero_ids,
            zero_times,

            one_data,
            one_distances,
            one_ids,
            one_times,
        )

    # --------------------------------------------------------
    # Verify IDs again after actual extraction
    # --------------------------------------------------------

    train_ids = set(
        outputs[
            "train"
        ][3].tolist()
    )

    val_ids = set(
        outputs[
            "val"
        ][3].tolist()
    )

    test_ids = set(
        outputs[
            "test"
        ][3].tolist()
    )

    assert train_ids.isdisjoint(
        val_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert val_ids.isdisjoint(
        test_ids
    )

    print(
        "\nLEAKAGE CHECK: PASSED"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    for split, result in outputs.items():

        save_split(
            split,
            *result,
        )

        print_split_summary(
            split,
            result[0],
            result[1],
            result[3],
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "version": "radar_v2",
        "description": (
            "Group-safe SENSE-X radar dataset. "
            "Complete ID_code experiments are isolated "
            "between train, validation and test."
        ),
        "seed": SEED,
        "max_samples_per_id": (
            MAX_SAMPLES_PER_ID
        ),
        "feature_channels": [
            "normalized_magnitude",
            "real",
            "imaginary",
            "phase",
        ],
        "num_range_bins": int(
            outputs[
                "train"
            ][0].shape[-1]
        ),
        "classes": {
            "0": "no_human",
            "1": "human",
        },
        "split_method": (
            "grouped_by_ID_code"
        ),
        "train_ids": sorted(
            train_ids
        ),
        "val_ids": sorted(
            val_ids
        ),
        "test_ids": sorted(
            test_ids
        ),
        "leakage_check": True,
    }

    with (
        OUT_DIR
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "RADAR V2 DATASET COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Output:",
        OUT_DIR,
    )

    print(
        "Existing radar dataset was NOT modified."
    )


if __name__ == "__main__":
    main()