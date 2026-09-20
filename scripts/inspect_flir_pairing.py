import json
from pathlib import Path


ROOT = Path(
    "data/raw/rgb/FLIR_ADAS_v2"
)


def inspect_json(path):

    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        "Type:",
        type(data).__name__
    )

    if isinstance(data, dict):

        print(
            "Keys:",
            list(data.keys())
        )

        for key, value in data.items():

            if isinstance(value, list):

                print(
                    f"{key}: list, length={len(value)}"
                )

                if len(value) > 0:

                    print(
                        "First item:"
                    )

                    print(
                        value[0]
                    )

            elif isinstance(value, dict):

                print(
                    f"{key}: dict, "
                    f"keys={list(value.keys())[:20]}"
                )

            else:

                print(
                    f"{key}: {value}"
                )


# ============================================================
# RGB INDEX
# ============================================================

rgb_index = (
    ROOT /
    "images_rgb_train" /
    "index.json"
)

thermal_index = (
    ROOT /
    "images_thermal_train" /
    "index.json"
)

rgb_coco = (
    ROOT /
    "images_rgb_train" /
    "coco.json"
)

thermal_coco = (
    ROOT /
    "images_thermal_train" /
    "coco.json"
)


print("=" * 70)
print("SENSE-X FLIR RGB / THERMAL PAIRING INSPECTION")
print("=" * 70)


for path in [
    rgb_index,
    thermal_index,
    rgb_coco,
    thermal_coco
]:

    if not path.exists():

        print(
            "\nMISSING:",
            path
        )

    else:

        print(
            "\nFOUND:",
            path
        )


# ============================================================
# INSPECT INDEX FILES
# ============================================================

if rgb_index.exists():
    inspect_json(rgb_index)

if thermal_index.exists():
    inspect_json(thermal_index)


# ============================================================
# INSPECT COCO IMAGE RECORDS
# ============================================================

def inspect_coco(path, name):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    images = data.get(
        "images",
        []
    )

    annotations = data.get(
        "annotations",
        []
    )

    categories = data.get(
        "categories",
        []
    )

    print(
        "Images:",
        len(images)
    )

    print(
        "Annotations:",
        len(annotations)
    )

    person_ids = {
        c["id"]
        for c in categories
        if c.get("name") == "person"
    }

    print(
        "Person category IDs:",
        person_ids
    )

    person_annotations = [
        a
        for a in annotations
        if a.get("category_id")
        in person_ids
    ]

    print(
        "Person annotations:",
        len(person_annotations)
    )

    print(
        "\nFirst 5 image records:"
    )

    for image in images[:5]:

        print(image)


    print(
        "\nFirst 5 person annotations:"
    )

    for annotation in person_annotations[:5]:

        print(annotation)

    return data


rgb_data = inspect_coco(
    rgb_coco,
    "RGB TRAIN COCO"
)

thermal_data = inspect_coco(
    thermal_coco,
    "THERMAL TRAIN COCO"
)


# ============================================================
# CHECK BASENAME OVERLAP
# ============================================================

if rgb_data and thermal_data:

    rgb_names = {
        Path(
            image["file_name"]
        ).name
        for image in rgb_data["images"]
    }

    thermal_names = {
        Path(
            image["file_name"]
        ).name
        for image in thermal_data["images"]
    }

    overlap = (
        rgb_names &
        thermal_names
    )

    print("\n" + "=" * 70)
    print("FILENAME OVERLAP")
    print("=" * 70)

    print(
        "RGB unique basenames:",
        len(rgb_names)
    )

    print(
        "Thermal unique basenames:",
        len(thermal_names)
    )

    print(
        "Exact basename overlap:",
        len(overlap)
    )

    if overlap:

        print(
            "\nExamples:"
        )

        for name in list(overlap)[:10]:

            print(
                " -",
                name
            )


print("\n" + "=" * 70)
print("PAIRING INSPECTION COMPLETE")
print("=" * 70)