from pathlib import Path
import sys

import cv2


BASE = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BASE)
)

from src.sensors.thermal_model import predict_thermal


# ============================================================
# TEST IMAGE
# ============================================================

image_path = (
    BASE
    / "data"
    / "processed"
    / "flir_yolo"
    / "images"
    / "thermal"
    / "val"
    / "pair_00592_frame_002040.png"
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(image_path),
    cv2.IMREAD_COLOR
)


if image is None:

    raise FileNotFoundError(
        f"Could not load:\n{image_path}"
    )


# ============================================================
# INFERENCE
# ============================================================

result = predict_thermal(
    image,
    conf_threshold=0.25
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("SENSE-X THERMAL INFERENCE TEST")
print("=" * 60)

print(
    f"\nImage: {image_path.name}"
)

print(
    f"Image shape: {image.shape}"
)

print(
    f"Detected persons: "
    f"{result['detections']}"
)

print(
    f"Human probability: "
    f"{result['human_probability']:.4f}"
)

print(
    f"Detected: "
    f"{result['detected']}"
)

print("\nDetections:")

for i, detection in enumerate(
    result["boxes"],
    start=1
):

    print(
        f"  Person {i}: "
        f"confidence="
        f"{detection['confidence']:.4f}, "
        f"bbox="
        f"{detection['bbox']}"
    )

print("\n" + "=" * 60)
print("THERMAL INFERENCE TEST COMPLETE")
print("=" * 60)