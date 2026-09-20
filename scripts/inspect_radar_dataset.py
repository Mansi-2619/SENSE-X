from pathlib import Path
import csv
import os

BASE = Path(__file__).resolve().parents[1]
RADAR_DIR = BASE / "data" / "raw" / "radar"


def inspect_file(path, name):
    print("=" * 70)
    print(f"INSPECTING {name}")
    print("=" * 70)

    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / (1024**3):.2f} GB")

    # ---------------------------------------------------------
    # 1. Read header only
    # ---------------------------------------------------------
    print("\n[1] HEADER")

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    print(f"Number of columns: {len(header)}")

    for i, col in enumerate(header[:30]):
        print(f"  [{i}] {col}")

    if len(header) > 30:
        print(f"  ... {len(header) - 30} more columns")

    # ---------------------------------------------------------
    # 2. Read only first 5 rows
    # ---------------------------------------------------------
    print("\n[2] FIRST 5 ROWS")

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)

        next(reader)  # skip header

        for row_num, row in enumerate(reader):
            if row_num >= 5:
                break

            print(f"\nRow {row_num + 1}:")
            print(f"  Number of values: {len(row)}")

            for i, value in enumerate(row[:15]):
                print(f"    {header[i]} = {value}")

            if len(row) > 15:
                print("    ...")

    # ---------------------------------------------------------
    # 3. Count rows WITHOUT loading them into memory
    # ---------------------------------------------------------
    print("\n[3] ROW COUNT")

    row_count = 0

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)

        next(reader)

        for _ in reader:
            row_count += 1

            if row_count % 100000 == 0:
                print(f"  scanned {row_count:,} rows...", end="\r")

    print(f"\nTotal data rows: {row_count:,}")

    # ---------------------------------------------------------
    # 4. Basic numeric statistics from first 1000 rows
    # ---------------------------------------------------------
    print("\n[4] SAMPLE TYPE CHECK")

    sample_rows = []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)

        next(reader)

        for i, row in enumerate(reader):
            if i >= 1000:
                break
            sample_rows.append(row)

    print(f"Sample rows inspected: {len(sample_rows):,}")

    # Determine whether values look numeric
    numeric_count = 0
    complex_count = 0

    for row in sample_rows:
        for value in row:
            value = value.strip()

            if not value:
                continue

            try:
                float(value)
                numeric_count += 1
            except ValueError:
                try:
                    complex(value.replace(" ", ""))
                    complex_count += 1
                except ValueError:
                    pass

    print(f"Numeric values detected: {numeric_count:,}")
    print(f"Complex values detected: {complex_count:,}")

    print("\n")


def main():

    n0 = RADAR_DIR / "N0.csv"
    n1 = RADAR_DIR / "N1.csv"

    print("=" * 70)
    print("SENSE-X RADAR DATASET — MEMORY-SAFE INSPECTION")
    print("=" * 70)

    print(f"\nRadar directory: {RADAR_DIR}")

    if not n0.exists():
        print(f"\nERROR: {n0} not found")

    if not n1.exists():
        print(f"\nERROR: {n1} not found")

    if n0.exists():
        inspect_file(n0, "N0 — NO HUMAN")

    if n1.exists():
        inspect_file(n1, "N1 — HUMAN PRESENT")

    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()