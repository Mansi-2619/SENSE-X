from pathlib import Path
import json
import csv


ROOT = Path("data/raw/rgb/FLIR_ADAS_v2")

RGB_INDEX = ROOT / "images_rgb_train" / "index.json"
THERMAL_INDEX = ROOT / "images_thermal_train" / "index.json"

PAIR_FILE = Path(
    "data/processed/flir_rgb_thermal_pairs.csv"
)

OUTPUT = Path(
    "data/processed/flir_person_manifest.csv"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_frame_key(frame):
    meta = frame.get("videoMetadata", {})
    return (
        str(meta.get("videoId")),
        int(meta.get("frameIndex"))
    )


def get_person_boxes(frame):
    boxes = []

    for ann in frame.get("annotations", []):

        labels = ann.get("labels", [])

        if "person" not in labels:
            continue

        box = ann.get("boundingBox")

        if not box:
            continue

        boxes.append({
            "x": box.get("x", 0),
            "y": box.get("y", 0),
            "w": box.get("w", 0),
            "h": box.get("h", 0)
        })

    return boxes


print("=" * 70)
print("SENSE-X FLIR PERSON DATASET BUILDER")
print("=" * 70)

print("\nLoading indexes...")

rgb_data = load_json(RGB_INDEX)
thermal_data = load_json(THERMAL_INDEX)

rgb_frames = {
    get_frame_key(f): f
    for f in rgb_data["frames"]
}

thermal_frames = {
    get_frame_key(f): f
    for f in thermal_data["frames"]
}

print("RGB frames:", len(rgb_frames))
print("Thermal frames:", len(thermal_frames))


print("\nLoading verified RGB↔thermal pairs...")

with open(
    PAIR_FILE,
    "r",
    encoding="utf-8"
) as f:

    pairs = list(csv.DictReader(f))


print("Verified pairs:", len(pairs))


rows = []

rgb_person_frames = 0
thermal_person_frames = 0
both_person_frames = 0

rgb_boxes_total = 0
thermal_boxes_total = 0


for pair in pairs:

    rgb_video = pair["rgb_video_id"]
    thermal_video = pair["thermal_video_id"]
    frame_index = int(pair["frame_index"])

    rgb_key = (
        rgb_video,
        frame_index
    )

    thermal_key = (
        thermal_video,
        frame_index
    )

    rgb_frame = rgb_frames.get(rgb_key)
    thermal_frame = thermal_frames.get(thermal_key)

    if rgb_frame is None or thermal_frame is None:
        continue

    rgb_boxes = get_person_boxes(rgb_frame)
    thermal_boxes = get_person_boxes(thermal_frame)

    if rgb_boxes:
        rgb_person_frames += 1

    if thermal_boxes:
        thermal_person_frames += 1

    if rgb_boxes and thermal_boxes:
        both_person_frames += 1

    rgb_boxes_total += len(rgb_boxes)
    thermal_boxes_total += len(thermal_boxes)

    rows.append({

        "rgb_path":
            pair["rgb_path"],

        "thermal_path":
            pair["thermal_path"],

        "rgb_video_id":
            rgb_video,

        "thermal_video_id":
            thermal_video,

        "frame_index":
            frame_index,

        "rgb_width":
            rgb_frame.get("width", ""),

        "rgb_height":
            rgb_frame.get("height", ""),

        "thermal_width":
            thermal_frame.get("width", ""),

        "thermal_height":
            thermal_frame.get("height", ""),

        "rgb_person_boxes":
            json.dumps(rgb_boxes),

        "thermal_person_boxes":
            json.dumps(thermal_boxes),

        "rgb_person_count":
            len(rgb_boxes),

        "thermal_person_count":
            len(thermal_boxes),

    })


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = list(rows[0].keys()) if rows else []


with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n" + "=" * 70)
print("PERSON DATASET SUMMARY")
print("=" * 70)

print("Processed pairs:", len(rows))
print(
    "Frames with RGB persons:",
    rgb_person_frames
)

print(
    "Frames with thermal persons:",
    thermal_person_frames
)

print(
    "Frames with persons in BOTH:",
    both_person_frames
)

print(
    "Total RGB person boxes:",
    rgb_boxes_total
)

print(
    "Total thermal person boxes:",
    thermal_boxes_total
)

print("\nManifest:")
print(OUTPUT)

print("\nDONE")