from pathlib import Path
import csv
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parent
RADAR_DIR = ROOT / "data" / "raw" / "radar"


def inspect_file(path: Path):
    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)

    ids = Counter()
    distances_by_id = defaultdict(Counter)

    total = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            total += 1

            id_code = row.get("ID_code", "").strip()

            ids[id_code] += 1

            distance_text = row.get("Distance", "").strip()

            if distance_text:
                try:
                    distance = round(float(distance_text), 3)
                    distances_by_id[id_code][distance] += 1
                except ValueError:
                    pass

    print(f"\nTOTAL ROWS: {total:,}")
    print(f"UNIQUE ID CODES: {len(ids):,}")

    print("\nUNIQUE EXPERIMENT IDS:\n")

    for id_code, count in sorted(ids.items()):
        distances = distances_by_id.get(id_code, {})

        distance_summary = (
            ", ".join(
                f"{distance}m ({n:,})"
                for distance, n in sorted(distances.items())
            )
            if distances
            else "--"
        )

        print(
            f"{id_code:<35} "
            f"rows={count:>9,}   "
            f"distance={distance_summary}"
        )


for filename in ["N0.csv", "N1.csv"]:
    inspect_file(RADAR_DIR / filename)