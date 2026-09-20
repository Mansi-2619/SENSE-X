import os
import json
from pathlib import Path


DATASET_DIR = Path(
    "data/raw/rgb"
)

print("=" * 70)
print("SENSE-X FLIR RGB + THERMAL DATASET INSPECTION")
print("=" * 70)

print("\nDataset directory:")
print(
    DATASET_DIR.resolve()
)


# ================================================================
# DIRECTORY CHECK
# ================================================================

if not DATASET_DIR.exists():

    print("\nERROR:")
    print(
        "FLIR dataset directory does not exist."
    )

    print(
        "\nExpected:"
    )

    print(
        "data/raw/rgb/"
    )

    raise SystemExit


# ================================================================
# FIND IMAGE FILES
# ================================================================

print("\nSearching for images...")

rgb_files = []
thermal_files = []
json_files = []

for root, dirs, files in os.walk(
    DATASET_DIR
):

    for filename in files:

        lower = filename.lower()

        full_path = os.path.join(
            root,
            filename
        )

        if lower.endswith(
            (".jpg", ".jpeg", ".png")
        ):

            # We will inspect naming later rather
            # than assuming every JPG is RGB.
            rgb_files.append(
                full_path
            )

        elif lower.endswith(
            (".tif", ".tiff")
        ):

            thermal_files.append(
                full_path
            )

        elif lower.endswith(
            ".json"
        ):

            json_files.append(
                full_path
            )


print(
    "\nImage files found:"
)

print(
    "JPEG/PNG:",
    len(rgb_files)
)

print(
    "TIFF:",
    len(thermal_files)
)

print(
    "JSON:",
    len(json_files)
)


# ================================================================
# SAMPLE FILES
# ================================================================

print("\n" + "=" * 70)
print("SAMPLE RGB/JPEG/PNG FILES")
print("=" * 70)

for file in rgb_files[:10]:

    print(
        " -",
        os.path.relpath(
            file,
            DATASET_DIR
        )
    )


print("\n" + "=" * 70)
print("SAMPLE THERMAL/TIFF FILES")
print("=" * 70)

for file in thermal_files[:10]:

    print(
        " -",
        os.path.relpath(
            file,
            DATASET_DIR
        )
    )


print("\n" + "=" * 70)
print("SAMPLE JSON FILES")
print("=" * 70)

for file in json_files[:10]:

    print(
        " -",
        os.path.relpath(
            file,
            DATASET_DIR
        )
    )


# ================================================================
# INSPECT ANNOTATION JSON
# ================================================================

if json_files:

    annotation_file = json_files[0]

    print("\n" + "=" * 70)
    print("INSPECTING ANNOTATION FILE")
    print("=" * 70)

    print(
        annotation_file
    )

    try:

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        print(
            "\nTop-level type:",
            type(data).__name__
        )

        if isinstance(data, dict):

            print(
                "\nTop-level keys:"
            )

            for key in data.keys():

                print(
                    " -",
                    key
                )

            if "images" in data:

                print(
                    "\nImages:",
                    len(data["images"])
                )

            if "annotations" in data:

                print(
                    "Annotations:",
                    len(data["annotations"])
                )

            if "categories" in data:

                print(
                    "Categories:"
                )

                for category in data[
                    "categories"
                ]:

                    print(
                        " -",
                        category
                    )

    except Exception as e:

        print(
            "\nCould not parse JSON:"
        )

        print(e)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)