from datasets import load_dataset
import numpy as np

DATASET = "ahlab-drone-project/DroneAudioSet"

print("Loading ground-truth audio...\n")

dataset = load_dataset(
    DATASET,
    "ground-truth",
    split="ref_001",
    streaming=True
)

for i, sample in enumerate(dataset):

    file_path = sample["file_path"]
    audio = sample["audio"]

    sr = audio["sampling_rate"]
    arr = np.asarray(audio["array"], dtype=np.float32)

    duration = len(arr) / sr

    print("=" * 60)
    print(f"FILE: {file_path}")
    print(f"Sampling rate: {sr}")
    print(f"Samples: {len(arr)}")
    print(f"Duration: {duration:.2f} sec")
    print(f"Min: {arr.min():.6f}")
    print(f"Max: {arr.max():.6f}")
    print(f"Mean: {arr.mean():.6f}")
    print(f"Std: {arr.std():.6f}")
    print(f"RMS: {np.sqrt(np.mean(arr ** 2)):.6f}")

    # Rough activity ratio
    threshold = 0.01
    active_ratio = np.mean(np.abs(arr) > threshold)

    print(f"Activity ratio (> {threshold}): {active_ratio:.4f}")

    if i >= 5:
        break