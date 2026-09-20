from pathlib import Path
import csv


class RadarRangeLookup:
    """
    Lightweight lookup for UniWA UWB radar distance metadata.

    This uses the dataset's recorded Distance field.
    It does NOT estimate range from the radar signal.
    """

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def get_first_human_distance(self):
        """
        Return the first valid distance associated with a human-present
        radar observation.
        """

        with self.csv_path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                try:
                    presence = int(float(row["Presence"]))
                    distance = float(row["Distance"])
                except (ValueError, TypeError, KeyError):
                    continue

                if presence == 1 and distance > 0:
                    return distance

        return None


def get_radar_range_lookup():
    return RadarRangeLookup(
        "data/raw/radar/N1.csv"
    )