from datasets import load_dataset
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

DATASET = "ahlab-drone-project/DroneAudioSet"

OUTPUT_DIR = "data/processed/audio/ground_truth_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    sr = int(audio["sampling_rate"])

    # Force waveform into a real 1-D NumPy array
    y = np.asarray(audio["array"], dtype=np.float32).squeeze()

    print("=" * 60)
    print(f"Analyzing: {file_path}")
    print(f"Original shape: {np.asarray(audio['array']).shape}")
    print(f"Processed shape: {y.shape}")
    print(f"Sampling rate: {sr}")
    print(f"Duration: {len(y) / sr:.2f} sec")

    if y.ndim != 1:
        print("WARNING: waveform is not 1-D. Skipping.")
        continue

    # Only analyze first 10 seconds
    segment_length = min(len(y), sr * 10)
    y_segment = y[:segment_length]

    print(f"Analyzing segment: {len(y_segment) / sr:.2f} sec")

    # Normalize safely
    peak = np.max(np.abs(y_segment))

    if peak > 0:
        y_segment = y_segment / peak

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y_segment,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    filename = os.path.basename(file_path).replace(".wav", "")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{filename}_spectrogram.png"
    )

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{filename} - Ground Truth Audio")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    # Additional acoustic features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y_segment,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y_segment,
        sr=sr
    )

    zero_crossing = librosa.feature.zero_crossing_rate(
        y_segment
    )

    rms = librosa.feature.rms(
        y=y_segment
    )

    print(
        f"Mean spectral centroid: "
        f"{np.mean(spectral_centroid):.2f} Hz"
    )

    print(
        f"Mean spectral bandwidth: "
        f"{np.mean(spectral_bandwidth):.2f} Hz"
    )

    print(
        f"Mean zero-crossing rate: "
        f"{np.mean(zero_crossing):.4f}"
    )

    print(
        f"Mean RMS: "
        f"{np.mean(rms):.6f}"
    )

    print(f"Saved: {output_path}")

    if i >= 5:
        break

print("\nAnalysis complete.")