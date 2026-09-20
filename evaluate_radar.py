from pathlib import Path

import numpy as np

from src.sensors.radar_model import get_radar_model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RADAR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "radar"
)

X_TEST_PATH = RADAR_DIR / "X_test.npy"
Y_TEST_PATH = RADAR_DIR / "y_test.npy"
DISTANCE_TEST_PATH = RADAR_DIR / "distance_test.npy"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SENSE-X UWB RADAR / NLOS EVALUATION")
print("=" * 70)

print("\nLoading radar test dataset...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
distance_test = np.load(DISTANCE_TEST_PATH)

print("X_test shape       :", X_test.shape)
print("y_test shape       :", y_test.shape)
print("distance_test shape:", distance_test.shape)

if not (
    len(X_test)
    == len(y_test)
    == len(distance_test)
):
    raise RuntimeError(
        "Radar test arrays do not have matching lengths."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained radar model...")

radar_model = get_radar_model()

print("Radar model loaded.")


# ============================================================
# INFERENCE
# ============================================================

probabilities = []
predictions = []

print("\nRunning inference on radar test set...\n")

for index in range(len(X_test)):

    result = radar_model.predict(
        X_test[index]
    )

    probability = float(
        result.get(
            "human_probability",
            result.get(
                "evidence",
                0.0
            )
        )
    )

    probability = max(
        0.0,
        min(
            1.0,
            probability
        )
    )

    prediction = (
        1
        if probability >= 0.50
        else 0
    )

    probabilities.append(
        probability
    )

    predictions.append(
        prediction
    )

    if (
        index % 100 == 0
        or index == len(X_test) - 1
    ):
        print(
            f"\rProcessed "
            f"{index + 1}/{len(X_test)}",
            end="",
            flush=True
        )


print("\n")

probabilities = np.asarray(
    probabilities,
    dtype=np.float64
)

predictions = np.asarray(
    predictions,
    dtype=np.int64
)

y_test = np.asarray(
    y_test,
    dtype=np.int64
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tp = int(
    np.sum(
        (predictions == 1)
        & (y_test == 1)
    )
)

tn = int(
    np.sum(
        (predictions == 0)
        & (y_test == 0)
    )
)

fp = int(
    np.sum(
        (predictions == 1)
        & (y_test == 0)
    )
)

fn = int(
    np.sum(
        (predictions == 0)
        & (y_test == 1)
    )
)


# ============================================================
# METRICS
# ============================================================

total = len(y_test)

accuracy = (
    (tp + tn) / total
    if total
    else 0.0
)

precision = (
    tp / (tp + fp)
    if (tp + fp)
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn)
    else 0.0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp)
    else 0.0
)

f1 = (
    2
    * precision
    * recall
    / (precision + recall)
    if (precision + recall)
    else 0.0
)


# ============================================================
# DATASET SUMMARY
# ============================================================

human_mask = (
    y_test == 1
)

no_human_mask = (
    y_test == 0
)

human_count = int(
    human_mask.sum()
)

no_human_count = int(
    no_human_mask.sum()
)


print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    "Total samples :",
    total
)

print(
    "Human samples :",
    human_count
)

print(
    "No-human     :",
    no_human_count
)


# ============================================================
# PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    f"""
                 PREDICTED

                HUMAN   EMPTY

ACTUAL HUMAN    {tp:<7} {fn:<7}
ACTUAL EMPTY    {fp:<7} {tn:<7}
"""
)


print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy     : {accuracy:.4f}"
)

print(
    f"Precision    : {precision:.4f}"
)

print(
    f"Recall       : {recall:.4f}"
)

print(
    f"Specificity  : {specificity:.4f}"
)

print(
    f"F1 Score     : {f1:.4f}"
)


# ============================================================
# PROBABILITY STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("HUMAN PROBABILITY STATISTICS")
print("=" * 70)


def print_probability_stats(
    name,
    values
):

    if len(values) == 0:
        print(
            name,
            ": no samples"
        )

        return

    print(
        f"{name:<12}"
        f"min={values.min():.6f}  "
        f"mean={values.mean():.6f}  "
        f"median={np.median(values):.6f}  "
        f"max={values.max():.6f}"
    )


print_probability_stats(
    "ALL",
    probabilities
)

print_probability_stats(
    "HUMAN",
    probabilities[
        human_mask
    ]
)

print_probability_stats(
    "NO HUMAN",
    probabilities[
        no_human_mask
    ]
)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("PROBABILITY DISTRIBUTION")
print("=" * 70)


bins = [
    (0.00, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 0.90),
    (0.90, 0.99),
    (0.99, 1.000001),
]


for low, high in bins:

    mask = (
        (probabilities >= low)
        & (probabilities < high)
    )

    count = int(
        mask.sum()
    )

    print(
        f"{low:>5.2f} - "
        f"{min(high, 1.0):>5.2f} : "
        f"{count:4d}"
    )


# ============================================================
# FALSE POSITIVES
# ============================================================

false_positive_indices = np.where(
    (predictions == 1)
    & (y_test == 0)
)[0]


print("\n")
print("=" * 70)
print("FALSE POSITIVES")
print("=" * 70)

if len(
    false_positive_indices
) == 0:

    print(
        "No false positives."
    )

else:

    sorted_fp = sorted(
        false_positive_indices,
        key=lambda i:
            probabilities[i],
        reverse=True
    )

    for index in sorted_fp[:20]:

        print(
            f"sample={index:<5} "
            f"actual=0 "
            f"prob={probabilities[index]:.6f}"
        )


# ============================================================
# FALSE NEGATIVES
# ============================================================

false_negative_indices = np.where(
    (predictions == 0)
    & (y_test == 1)
)[0]


print("\n")
print("=" * 70)
print("FALSE NEGATIVES")
print("=" * 70)

if len(
    false_negative_indices
) == 0:

    print(
        "No false negatives."
    )

else:

    sorted_fn = sorted(
        false_negative_indices,
        key=lambda i:
            probabilities[i]
    )

    for index in sorted_fn[:20]:

        distance = (
            float(
                distance_test[index]
            )
            if np.isfinite(
                distance_test[index]
            )
            else None
        )

        print(
            f"sample={index:<5} "
            f"actual=1 "
            f"prob={probabilities[index]:.6f} "
            f"distance={distance}"
        )


# ============================================================
# RANGE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("THROUGH-WALL HUMAN RANGE ANALYSIS")
print("=" * 70)

human_distances = (
    distance_test[
        human_mask
    ]
)

valid_human_distances = (
    human_distances[
        np.isfinite(
            human_distances
        )
    ]
)

if len(
    valid_human_distances
):

    print(
        "Valid human ranges:",
        len(
            valid_human_distances
        )
    )

    print(
        "Minimum range:",
        float(
            valid_human_distances.min()
        ),
        "m"
    )

    print(
        "Maximum range:",
        float(
            valid_human_distances.max()
        ),
        "m"
    )

    print(
        "Mean range:",
        float(
            valid_human_distances.mean()
        ),
        "m"
    )


    unique_ranges = np.unique(
        np.round(
            valid_human_distances,
            2
        )
    )

    print(
        "\nObserved ranges:"
    )

    for distance in unique_ranges:

        indices = np.where(
            human_mask
            & np.isfinite(
                distance_test
            )
            & (
                np.round(
                    distance_test,
                    2
                )
                == distance
            )
        )[0]

        if not len(indices):
            continue

        range_accuracy = float(
            np.mean(
                predictions[
                    indices
                ]
                == y_test[
                    indices
                ]
            )
        )

        mean_probability = float(
            np.mean(
                probabilities[
                    indices
                ]
            )
        )

        print(
            f"{distance:.2f} m"
            f"  samples={len(indices):4d}"
            f"  detection={range_accuracy:.4f}"
            f"  mean_prob={mean_probability:.4f}"
        )


# ============================================================
# SAMPLE INSPECTION
# ============================================================

print("\n")
print("=" * 70)
print("FIRST 30 TEST SAMPLES")
print("=" * 70)

for index in range(
    min(
        30,
        total
    )
):

    distance = (
        f"{float(distance_test[index]):.2f} m"
        if np.isfinite(
            distance_test[index]
        )
        else "--"
    )

    print(
        f"{index:4d} | "
        f"actual={int(y_test[index])} | "
        f"pred={int(predictions[index])} | "
        f"prob={probabilities[index]:.6f} | "
        f"range={distance}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    PROJECT_ROOT
    / "radar_evaluation_results.csv"
)

result_matrix = np.column_stack(
    (
        np.arange(total),
        y_test,
        predictions,
        probabilities,
        distance_test
    )
)

np.savetxt(
    output_path,
    result_matrix,
    delimiter=",",
    header=(
        "sample_index,"
        "actual_label,"
        "predicted_label,"
        "human_probability,"
        "distance_m"
    ),
    comments="",
    fmt=[
        "%d",
        "%d",
        "%d",
        "%.8f",
        "%.6f"
    ]
)


print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "Detailed results saved to:"
)

print(
    output_path
)

print("=" * 70)