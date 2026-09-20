from datasets import load_dataset

DATASET = "ahlab-drone-project/DroneAudioSet"

print("Loading DroneAudioSet metadata...")

dataset = load_dataset(
    DATASET,
    "drone-with-source",
    split="train_001",
    streaming=True
)

print("Dataset loaded.\n")

for i, sample in enumerate(dataset):

    print(f"--- SAMPLE {i + 1} ---")

    print("file_path:", sample["file_path"])
    print("data_type:", sample["data_type"])

    audio = sample["audio"]

    print("sampling_rate:", audio["sampling_rate"])

    audio_array = audio["array"]

    print("audio type:", type(audio_array))
    print("number of samples:", len(audio_array))

    if len(audio_array) > 0:
        print("first 5 samples:", audio_array[:5])

    duration = len(audio_array) / audio["sampling_rate"]

    print("duration:", round(duration, 2), "seconds")
    print()

    if i >= 4:
        break