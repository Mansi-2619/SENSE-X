import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.localization.radar_range import get_radar_range_lookup


def main():
    print("=" * 70)
    print("SENSE-X — RADAR RANGE METADATA TEST")
    print("=" * 70)

    lookup = get_radar_range_lookup()

    distance = lookup.get_first_human_distance()

    if distance is None:
        print("No valid human-present distance found.")
    else:
        print(f"Dataset radar distance: {distance:.2f} m")

    print("=" * 70)


if __name__ == "__main__":
    main()