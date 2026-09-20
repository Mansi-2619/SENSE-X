from datasets import load_dataset

DATASET = "ahlab-drone-project/DroneAudioSet"

print("Loading ground-truth metadata...")

dataset = load_dataset(
    DATASET,
    "ground-truth",
    split="ref_001",
    streaming=True
)

print("Dataset loaded.\n")

for i, sample in enumerate(dataset):

    print(f"--- RECORD {i + 1} ---")

    for key, value in sample.items():

        if key == "audio":
            print("audio: <audio data>")
        else:
            print(f"{key}: {value}")

    print()

    if i >= 9:
        break