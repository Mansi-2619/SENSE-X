from pathlib import Path

import numpy as np
import soundfile as sf

from src.sensors.audio_model import get_audio_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# Find a local audio file
# ------------------------------------------------------------

AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "audio"
)


audio_files = list(
    AUDIO_DIR.rglob("*.wav")
)

if not audio_files:

    audio_files = list(
        AUDIO_DIR.rglob("*.flac")
    )

if not audio_files:

    raise FileNotFoundError(
        "No .wav or .flac audio file found under "
        f"{AUDIO_DIR}"
    )


audio_path = audio_files[0]


print("=" * 60)
print("SENSE-X AUDIO INFERENCE TEST")
print("=" * 60)

print()
print("Audio:", audio_path)


# ------------------------------------------------------------
# Load audio
# ------------------------------------------------------------

audio, sample_rate = sf.read(
    audio_path,
    dtype="float32"
)

print(
    "Original sample rate:",
    sample_rate
)

print(
    "Audio shape:",
    audio.shape
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print()
print("Loading audio model...")

model = get_audio_model()

print("Audio model loaded successfully.")


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

print()
print("Running inference...")

result = model.predict(
    audio,
    sample_rate=sample_rate,
    number_of_clips=3
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 60)
print("AUDIO INFERENCE RESULT")
print("=" * 60)

print(
    f"Source-present probability : "
    f"{result['source_present_probability']:.4f}"
)

print(
    f"Drone-only probability    : "
    f"{result['drone_only_probability']:.4f}"
)

print(
    "Prediction                :",
    result["prediction"]
)

print(
    "Label                     :",
    result["label"]
)

print(
    "Clips analyzed            :",
    result["clips_analyzed"]
)

print()
print("Per-clip results:")

for index, clip_result in enumerate(
    result["clip_predictions"],
    start=1
):

    print(
        f"  Clip {index}: "
        f"source={clip_result['source_present_probability']:.4f}, "
        f"drone={clip_result['drone_only_probability']:.4f}, "
        f"label={clip_result['label']}"
    )

print()
print("=" * 60)
print("AUDIO INFERENCE TEST COMPLETE")
print("=" * 60)