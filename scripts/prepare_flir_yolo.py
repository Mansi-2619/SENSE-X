from pathlib import Path
import ast
import csv
import gc
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import tifffile


BASE = Path(__file__).resolve().parents[1]

MANIFEST = BASE / "data" / "processed" / "flir_person_manifest.csv"
OUT = BASE / "data" / "processed" / "flir_yolo"

TRAIN_PAIRS = 2400
VAL_PAIRS = 600


def parse_boxes(value):
    """Parse JSON/list of FLIR person boxes."""

    if pd.isna(value) or not str(value).strip():
        return []

    value = str(value).strip()

    try:
        boxes = json.loads(value)
    except Exception:
        boxes = ast.literal_eval(value)

    if not isinstance(boxes, list):
        raise ValueError(
            f"Expected list of boxes, got {type(boxes)}"
        )

    return boxes


def yolo_line(box, width, height):
    """
    Convert:
        {"x": x, "y": y, "w": w, "h": h}

    into:
        class x_center y_center width height
    """

    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])

    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height

    w_norm = w / width
    h_norm = h / height

    # Clamp for YOLO safety
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return (
        f"0 {x_center:.6f} "
        f"{y_center:.6f} "
        f"{w_norm:.6f} "
        f"{h_norm:.6f}"
    )


def write_labels(boxes, width, height, destination):

    with open(
        destination,
        "w",
        encoding="utf-8",
    ) as f:

        for box in boxes:

            f.write(
                yolo_line(
                    box,
                    width,
                    height,
                )
                + "\n"
            )


def thermal_to_png(src, dst):
    """
    Convert 16-bit LZW thermal TIFF to compact 8-bit PNG.

    Uses integer scaling to avoid unnecessary float32
    allocations.
    """

    arr = tifffile.imread(src)

    if arr.ndim > 2:
        arr = np.squeeze(arr)

    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)

    # Percentile calculation uses only a tiny temporary array.
    lo, hi = np.percentile(
        arr,
        [2, 98],
    )

    lo = int(max(0, min(65535, lo)))
    hi = int(max(lo + 1, min(65535, hi)))

    # Integer arithmetic rather than float32 normalization.
    work = arr.astype(np.uint32)

    work = np.clip(
        work,
        lo,
        hi,
    )

    work = (
        (work - lo) * 255 // (hi - lo)
    ).astype(np.uint8)

    Image.fromarray(work).save(
        dst,
        format="PNG",
    )

    del work
    del arr

    gc.collect()


def ensure_dirs():

    for split in ["train", "val"]:

        for modality in ["rgb", "thermal"]:

            (OUT / "images" / modality / split).mkdir(
                parents=True,
                exist_ok=True,
            )

            (OUT / "labels" / modality / split).mkdir(
                parents=True,
                exist_ok=True,
            )


def process_pair(row, split, index):

    rgb_src = BASE / str(row["rgb_path"])
    thermal_src = BASE / str(row["thermal_path"])

    stem = (
        f"pair_{index:05d}_"
        f"frame_{int(row['frame_index']):06d}"
    )

    rgb_dst = (
        OUT
        / "images"
        / "rgb"
        / split
        / f"{stem}.jpg"
    )

    thermal_dst = (
        OUT
        / "images"
        / "thermal"
        / split
        / f"{stem}.png"
    )

    rgb_label = (
        OUT
        / "labels"
        / "rgb"
        / split
        / f"{stem}.txt"
    )

    thermal_label = (
        OUT
        / "labels"
        / "thermal"
        / split
        / f"{stem}.txt"
    )

    rgb_boxes = parse_boxes(
        row["rgb_person_boxes"]
    )

    thermal_boxes = parse_boxes(
        row["thermal_person_boxes"]
    )

    # ----------------------------------------------------------
    # RGB
    # ----------------------------------------------------------

    if not rgb_dst.exists():

        shutil.copy2(
            rgb_src,
            rgb_dst,
        )

    if not rgb_label.exists():

        write_labels(
            rgb_boxes,
            int(row["rgb_width"]),
            int(row["rgb_height"]),
            rgb_label,
        )

    # ----------------------------------------------------------
    # THERMAL
    # ----------------------------------------------------------

    if not thermal_dst.exists():

        thermal_to_png(
            thermal_src,
            thermal_dst,
        )

    if not thermal_label.exists():

        write_labels(
            thermal_boxes,
            int(row["thermal_width"]),
            int(row["thermal_height"]),
            thermal_label,
        )

    gc.collect()


def process_split(rows, split):

    print()
    print("=" * 70)
    print(f"PROCESSING {split.upper()}")
    print("=" * 70)

    success = 0
    failed = 0

    for i, (_, row) in enumerate(rows.iterrows()):

        try:

            process_pair(
                row,
                split,
                i,
            )

            success += 1

            if success % 100 == 0:
                print(
                    f"{split}: "
                    f"{success}/{len(rows)}"
                )

        except Exception as e:

            failed += 1

            print(
                f"[WARN] {split} item {i}: "
                f"{type(e).__name__}: {e}"
            )

            # Don't kill the entire dataset.
            continue

    print()
    print(
        f"{split} complete: "
        f"{success} successful, "
        f"{failed} failed"
    )

    return success, failed


def write_yaml(modality):

    yaml_path = OUT / f"{modality}.yaml"

    content = f"""path: {OUT.as_posix()}

train: images/{modality}/train
val: images/{modality}/val

names:
  0: person
"""

    yaml_path.write_text(
        content,
        encoding="utf-8",
    )


def main():

    print("=" * 70)
    print("SENSE-X FLIR YOLO PREPARATION")
    print("=" * 70)

    print(f"Manifest: {MANIFEST}")
    print(f"Output:   {OUT}")

    ensure_dirs()

    df = pd.read_csv(
        MANIFEST
    )

    # Only synchronized pairs where BOTH sensors
    # have at least one person annotation.
    df = df[
        (df["rgb_person_count"] > 0)
        &
        (df["thermal_person_count"] > 0)
    ].copy()

    print()
    print(
        f"Synchronized human pairs available: "
        f"{len(df)}"
    )

    # Deterministic selection
    df = df.sample(
        frac=1,
        random_state=42,
    ).reset_index(
        drop=True
    )

    total_needed = (
        TRAIN_PAIRS + VAL_PAIRS
    )

    if len(df) < total_needed:

        total_needed = len(df)

        TRAIN_PAIRS_ACTUAL = int(
            total_needed * 0.8
        )

        VAL_PAIRS_ACTUAL = (
            total_needed
            - TRAIN_PAIRS_ACTUAL
        )

    else:

        TRAIN_PAIRS_ACTUAL = TRAIN_PAIRS
        VAL_PAIRS_ACTUAL = VAL_PAIRS

    train_rows = df.iloc[
        :TRAIN_PAIRS_ACTUAL
    ].copy()

    val_rows = df.iloc[
        TRAIN_PAIRS_ACTUAL:
        TRAIN_PAIRS_ACTUAL + VAL_PAIRS_ACTUAL
    ].copy()

    print(
        f"Train pairs: {len(train_rows)}"
    )

    print(
        f"Validation pairs: {len(val_rows)}"
    )

    # ----------------------------------------------------------
    # Process
    # ----------------------------------------------------------

    train_success, train_failed = process_split(
        train_rows,
        "train",
    )

    val_success, val_failed = process_split(
        val_rows,
        "val",
    )

    # ----------------------------------------------------------
    # YAML
    # ----------------------------------------------------------

    write_yaml("rgb")
    write_yaml("thermal")

    # ----------------------------------------------------------
    # Final
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("FLIR PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"Train: "
        f"{train_success} successful / "
        f"{len(train_rows)}"
    )

    print(
        f"Validation: "
        f"{val_success} successful / "
        f"{len(val_rows)}"
    )

    print(
        f"Total successful pairs: "
        f"{train_success + val_success}"
    )

    print()
    print("Created:")
    print(
        f"  {OUT / 'rgb.yaml'}"
    )
    print(
        f"  {OUT / 'thermal.yaml'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()