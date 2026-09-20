import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from datasets import load_dataset

from src.sensors.audio_model import get_audio_model


def main():
    print("=" * 60)
    print("SENSE-X AUDIO MODEL — STREAMING DATASET TEST")
    print("=" * 60)

    print("\nLoading one DroneAudioSet sample...")

    ds = load_dataset(
        "ahlab-drone-project/DroneAudioSet",
        "drone-with-source",
        split="train_001",
        streaming=True,
    )

    sample = next(iter(ds))

    audio = sample["audio"]
    audio_array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = audio["sampling_rate"]

    print(f"\nFile: {sample['file_path']}")
    print(f"Data type: {sample['data_type']}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Audio shape: {audio_array.shape}")
    print(f"Audio duration: {len(audio_array) / sampling_rate:.2f} sec")

    print("\nLoading SENSE-X audio model...")

    model = get_audio_model()

    result = model.predict(
        audio_array,
        sample_rate=sampling_rate,
        number_of_clips=3,
    )

    print("\n" + "-" * 60)
    print("AUDIO INFERENCE RESULT")
    print("-" * 60)

    print(
        f"Source-present evidence: "
        f"{result['source_present_probability']:.4f}"
    )

    print(
        f"Drone-only evidence:     "
        f"{result['drone_only_probability']:.4f}"
    )

    print(f"Prediction:              {result['prediction']}")
    print(f"Label:                   {result['label']}")
    print(f"Clips analyzed:          {result['clips_analyzed']}")

    print("\nPer-clip results:")

    for i, clip in enumerate(result["clip_predictions"], start=1):
        print(
            f"  Clip {i}: "
            f"source={clip['source_present_probability']:.4f}, "
            f"drone={clip['drone_only_probability']:.4f}, "
            f"prediction={clip['prediction']}, "
            f"label={clip['label']}"
        )

    print("\n" + "=" * 60)
    print("AUDIO STREAMING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()