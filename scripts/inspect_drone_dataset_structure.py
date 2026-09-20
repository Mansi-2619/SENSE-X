from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

DATASET = "ahlab-drone-project/DroneAudioSet"

print("=" * 70)
print("DRONE AUDIO DATASET STRUCTURE")
print("=" * 70)

configs = get_dataset_config_names(DATASET)

print("\nCONFIGURATIONS:")
for config in configs:
    print(" -", config)

print("\n" + "=" * 70)

for config in configs:

    print(f"\nCONFIG: {config}")

    try:
        splits = get_dataset_split_names(
            DATASET,
            config_name=config
        )

        print("SPLITS:")
        for split in splits:
            print(" -", split)

    except Exception as e:
        print("Could not inspect splits:", e)

print("\n" + "=" * 70)
print("INSPECTING FIRST SAMPLE OF EACH CONFIG")
print("=" * 70)

for config in configs:

    print(f"\n### {config}")

    try:
        splits = get_dataset_split_names(
            DATASET,
            config_name=config
        )

        split = splits[0]

        ds = load_dataset(
            DATASET,
            config,
            split=split,
            streaming=True
        )

        sample = next(iter(ds))

        print("Split:", split)
        print("Keys:", list(sample.keys()))

        for key, value in sample.items():

            if key == "audio":
                print("audio: <audio object>")

            else:
                print(f"{key}: {value}")

    except Exception as e:
        print("ERROR:", e)

print("\nInspection complete.")