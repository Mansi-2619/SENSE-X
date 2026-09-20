from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RADAR_DIR = Path(
    "data/raw/radar"
)

NROWS = 1000


# ============================================================
# FILES
# ============================================================

N0_PATH = (
    RADAR_DIR / "N0.csv"
)

N1_PATH = (
    RADAR_DIR / "N1.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SENSE-X RADAR DATASET")
print("MEMORY-SAFE INSPECTION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    N0_PATH,
    N1_PATH
]:

    if not path.exists():

        print(
            f"\nERROR: File not found:"
        )

        print(
            path.resolve()
        )

        raise SystemExit(1)


# ============================================================
# INSPECTION FUNCTION
# ============================================================

def inspect_file(
    path,
    name
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    # --------------------------------------------------------
    # FILE SIZE
    # --------------------------------------------------------

    size_bytes = (
        path.stat().st_size
    )

    size_gb = (
        size_bytes /
        (1024 ** 3)
    )

    print(
        "\nFile:",
        path
    )

    print(
        "File size:",
        f"{size_gb:.2f} GB"
    )

    # --------------------------------------------------------
    # READ ONLY FIRST 1000 ROWS
    # --------------------------------------------------------

    print(
        f"\nReading first {NROWS} rows..."
    )

    try:

        df = pd.read_csv(
            path,
            nrows=NROWS
        )

    except Exception as e:

        print(
            "\nERROR while reading:"
        )

        print(e)

        return

    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    print(
        "\nChunk shape:",
        df.shape
    )

    print(
        "Number of columns:",
        len(df.columns)
    )

    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "COLUMNS"
    )

    print(
        "-" * 70
    )

    for index, column in enumerate(
        df.columns
    ):

        print(
            f"{index:5d}: {column}"
        )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "DATA TYPES"
    )

    print(
        "-" * 70
    )

    print(
        df.dtypes
    )

    # --------------------------------------------------------
    # FIRST 5 ROWS
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "FIRST 5 ROWS"
    )

    print(
        "-" * 70
    )

    print(
        df.head().to_string()
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "MISSING VALUES"
    )

    print(
        "-" * 70
    )

    missing = (
        df.isna().sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print(
            "No missing values in first "
            f"{NROWS} rows."
        )

    else:

        print(
            missing
        )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory_mb = (
        df.memory_usage(
            deep=True
        ).sum()
        / (1024 ** 2)
    )

    print(
        "\nChunk memory:",
        f"{memory_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "FIRST ROW VALUES"
    )

    print(
        "-" * 70
    )

    if len(df) > 0:

        for column in df.columns:

            value = df.iloc[
                0
            ][column]

            print(
                f"{column}: {value}"
            )

    # --------------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "NUMERIC SUMMARY"
    )

    print(
        "-" * 70
    )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    if len(numeric_columns) > 0:

        print(
            df[
                numeric_columns
            ].describe().T
        )

    else:

        print(
            "No numeric columns detected."
        )


# ============================================================
# N0
# ============================================================

inspect_file(
    N0_PATH,
    "N0 — NO HUMAN"
)


# ============================================================
# N1
# ============================================================

inspect_file(
    N1_PATH,
    "N1 — HUMAN PRESENT"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("RADAR INSPECTION COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "The complete 18+ GB files were NOT loaded."
)

print(
    f"Only the first {NROWS} rows of each file were inspected."
)