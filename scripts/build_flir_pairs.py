from pathlib import Path
import json
import csv
import re


# ============================================================
# PATHS
# ============================================================

ROOT = Path("data/raw/rgb/FLIR_ADAS_v2")

RGB_ROOT = ROOT / "images_rgb_train"
THERMAL_ROOT = ROOT / "images_thermal_train"

RGB_INDEX = RGB_ROOT / "index.json"
THERMAL_INDEX = THERMAL_ROOT / "index.json"

OUTPUT = Path(
    "data/processed/flir_rgb_thermal_pairs.csv"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FRAME METADATA
# ============================================================

def get_frame_metadata(frame):

    metadata = frame.get("videoMetadata", {})

    video_id = metadata.get("videoId")
    frame_index = metadata.get("frameIndex")

    if video_id is None or frame_index is None:
        return None, None

    return str(video_id), int(frame_index)


# ============================================================
# METADATA LOOKUP
# ============================================================

def build_frame_lookup(frames):

    lookup = {}

    for frame in frames:

        video_id, frame_index = \
            get_frame_metadata(frame)

        if video_id is None:
            continue

        lookup[
            (video_id, frame_index)
        ] = frame

    return lookup


# ============================================================
# THERMAL → RGB VIDEO MAPPING
# ============================================================

def extract_rgb_video_id(description):

    if not description:
        return None

    match = re.search(
        r'"RGB"\s*:\s*"([^"]+)"',
        description
    )

    if match:
        return match.group(1)

    return None


def build_video_mapping(thermal_videos):

    mapping = {}

    for video in thermal_videos:

        thermal_id = video.get("id")

        if thermal_id is None:
            continue

        rgb_id = extract_rgb_video_id(
            video.get("description", "")
        )

        if rgb_id:

            mapping[
                str(thermal_id)
            ] = str(rgb_id)

    return mapping


# ============================================================
# ACTUAL IMAGE FILE INDEX
# ============================================================

def index_image_files(root):

    """
    Build:

        (video_id, frame_index) -> actual file path

    from filenames such as:

        video-23bsd9bsr962GdFBZ-frame-000221-xxxx.jpg
    """

    lookup = {}

    pattern = re.compile(
        r"^video-(.+)-frame-(\d+)-.+\.(jpg|jpeg|png|tif|tiff)$",
        re.IGNORECASE
    )

    total_files = 0
    matched_files = 0

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        }:
            continue

        total_files += 1

        match = pattern.match(path.name)

        if not match:
            continue

        video_id = match.group(1)
        frame_index = int(match.group(2))

        key = (
            video_id,
            frame_index
        )

        # Keep the first if duplicate physical files exist.
        if key not in lookup:

            lookup[key] = path

        matched_files += 1

    return lookup, total_files, matched_files


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SENSE-X FLIR RGB ↔ THERMAL PAIR BUILDER")
print("=" * 70)


# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

print("\nLoading RGB index...")

rgb_data = load_json(RGB_INDEX)

print("Loading thermal index...")

thermal_data = load_json(THERMAL_INDEX)


rgb_frames = rgb_data["frames"]
thermal_frames = thermal_data["frames"]

rgb_videos = rgb_data["videos"]
thermal_videos = thermal_data["videos"]


print(f"\nRGB frames: {len(rgb_frames)}")
print(f"Thermal frames: {len(thermal_frames)}")

print(f"RGB videos: {len(rgb_videos)}")
print(f"Thermal videos: {len(thermal_videos)}")


# ------------------------------------------------------------
# Metadata lookups
# ------------------------------------------------------------

print("\nBuilding RGB frame lookup...")

rgb_lookup = build_frame_lookup(
    rgb_frames
)

print(
    f"RGB lookup entries: "
    f"{len(rgb_lookup)}"
)


print("\nBuilding thermal frame lookup...")

thermal_lookup = build_frame_lookup(
    thermal_frames
)

print(
    f"Thermal lookup entries: "
    f"{len(thermal_lookup)}"
)


# ------------------------------------------------------------
# Video mapping
# ------------------------------------------------------------

print(
    "\nBuilding thermal → RGB video mapping..."
)

thermal_to_rgb = build_video_mapping(
    thermal_videos
)

print(
    f"Thermal videos with RGB mapping: "
    f"{len(thermal_to_rgb)}"
)


print("\nExample mappings:")

for thermal_id, rgb_id in list(
    thermal_to_rgb.items()
)[:10]:

    print(
        f"  {thermal_id} -> {rgb_id}"
    )


# ------------------------------------------------------------
# Physical RGB files
# ------------------------------------------------------------

print(
    "\nIndexing actual RGB image files..."
)

rgb_file_lookup, rgb_total, rgb_matched = \
    index_image_files(RGB_ROOT)

print(
    f"RGB image files found: "
    f"{rgb_total}"
)

print(
    f"RGB files matching FLIR naming pattern: "
    f"{rgb_matched}"
)

print(
    f"RGB video/frame file entries: "
    f"{len(rgb_file_lookup)}"
)


# ------------------------------------------------------------
# Physical thermal files
# ------------------------------------------------------------

print(
    "\nIndexing actual thermal image files..."
)

thermal_file_lookup, thermal_total, thermal_matched = \
    index_image_files(THERMAL_ROOT)

print(
    f"Thermal image files found: "
    f"{thermal_total}"
)

print(
    f"Thermal files matching FLIR naming pattern: "
    f"{thermal_matched}"
)

print(
    f"Thermal video/frame file entries: "
    f"{len(thermal_file_lookup)}"
)


# ============================================================
# BUILD PAIRS
# ============================================================

print("\nBuilding frame pairs...")


pairs = []

missing_mapping = 0
missing_rgb_frame_metadata = 0
missing_rgb_file = 0
missing_thermal_file = 0


for thermal_frame in thermal_frames:

    thermal_video_id, frame_index = \
        get_frame_metadata(thermal_frame)

    if thermal_video_id is None:
        continue

    # --------------------------------------------------------
    # Thermal video -> RGB video
    # --------------------------------------------------------

    rgb_video_id = thermal_to_rgb.get(
        thermal_video_id
    )

    if rgb_video_id is None:

        missing_mapping += 1
        continue

    # --------------------------------------------------------
    # RGB metadata frame
    # --------------------------------------------------------

    rgb_key = (
        rgb_video_id,
        frame_index
    )

    rgb_frame = rgb_lookup.get(
        rgb_key
    )

    if rgb_frame is None:

        missing_rgb_frame_metadata += 1
        continue

    # --------------------------------------------------------
    # Actual RGB file
    # --------------------------------------------------------

    rgb_path = rgb_file_lookup.get(
        rgb_key
    )

    if rgb_path is None:

        missing_rgb_file += 1
        continue

    # --------------------------------------------------------
    # Actual thermal file
    # --------------------------------------------------------

    thermal_key = (
        thermal_video_id,
        frame_index
    )

    thermal_path = thermal_file_lookup.get(
        thermal_key
    )

    if thermal_path is None:

        missing_thermal_file += 1
        continue

    # --------------------------------------------------------
    # Save pair
    # --------------------------------------------------------

    pairs.append({

        "thermal_video_id":
            thermal_video_id,

        "rgb_video_id":
            rgb_video_id,

        "frame_index":
            frame_index,

        "rgb_path":
            str(rgb_path).replace(
                "\\", "/"
            ),

        "thermal_path":
            str(thermal_path).replace(
                "\\", "/"
            ),

        "rgb_dataset_frame_id":
            rgb_frame.get(
                "datasetFrameId",
                ""
            ),

        "thermal_dataset_frame_id":
            thermal_frame.get(
                "datasetFrameId",
                ""
            ),

        "rgb_width":
            rgb_frame.get(
                "width",
                ""
            ),

        "rgb_height":
            rgb_frame.get(
                "height",
                ""
            ),

        "thermal_width":
            thermal_frame.get(
                "width",
                ""
            ),

        "thermal_height":
            thermal_frame.get(
                "height",
                ""
            ),

    })


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PAIRING RESULTS")
print("=" * 70)

print(
    f"Thermal → RGB video mappings: "
    f"{len(thermal_to_rgb)}"
)

print(
    f"Valid RGB ↔ thermal pairs: "
    f"{len(pairs)}"
)

print(
    f"Missing thermal → RGB video mapping: "
    f"{missing_mapping}"
)

print(
    f"Missing RGB frame metadata: "
    f"{missing_rgb_frame_metadata}"
)

print(
    f"Missing RGB image file: "
    f"{missing_rgb_file}"
)

print(
    f"Missing thermal image file: "
    f"{missing_thermal_file}"
)


# ============================================================
# SAVE MANIFEST
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = [
    "thermal_video_id",
    "rgb_video_id",
    "frame_index",
    "rgb_path",
    "thermal_path",
    "rgb_dataset_frame_id",
    "thermal_dataset_frame_id",
    "rgb_width",
    "rgb_height",
    "thermal_width",
    "thermal_height",
]


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
    writer.writerows(pairs)


print("\nManifest saved:")
print(OUTPUT)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE VERIFIED PAIRS")
print("=" * 70)


for pair in pairs[:10]:

    print(
        f"\nFrame index: "
        f"{pair['frame_index']}"
    )

    print(
        f"RGB: "
        f"{pair['rgb_path']}"
    )

    print(
        f"Thermal: "
        f"{pair['thermal_path']}"
    )

    print(
        f"RGB video: "
        f"{pair['rgb_video_id']}"
    )

    print(
        f"Thermal video: "
        f"{pair['thermal_video_id']}"
    )

    print(
        f"RGB size: "
        f"{pair['rgb_width']} x "
        f"{pair['rgb_height']}"
    )

    print(
        f"Thermal size: "
        f"{pair['thermal_width']} x "
        f"{pair['thermal_height']}"
    )


print("\n" + "=" * 70)
print("PAIR BUILD COMPLETE")
print("=" * 70)