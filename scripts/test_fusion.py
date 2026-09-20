from src.fusion.oasf_engine import fuse_sensors


def print_result(name, result):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Human score : {result.human_score:.4f}")
    print(f"Uncertainty : {result.uncertainty:.4f}")
    print(f"Decision    : {result.decision}")

    print("\nEvidence:")
    print(f"  Thermal : {result.thermal_evidence:.4f}")
    print(f"  Radar   : {result.radar_evidence:.4f}")
    print(f"  Audio   : {result.audio_evidence:.4f}")

    print("\nAdaptive weights:")
    print(f"  Thermal : {result.thermal_weight:.4f}")
    print(f"  Radar   : {result.radar_weight:.4f}")
    print(f"  Audio   : {result.audio_weight:.4f}")

    print("\nExplanation:")
    print(f"  {result.explanation}")


def main():

    # Scenario 1:
    # Clear environment.
    clear = fuse_sensors(
        thermal_evidence=0.87,
        radar_evidence=0.72,
        audio_evidence=0.41,
        occlusion=0.10,
        noise=0.10,
    )

    print_result(
        "SCENARIO 1 — CLEAR ENVIRONMENT",
        clear,
    )

    # Scenario 2:
    # Heavy rubble / visual obstruction.
    rubble = fuse_sensors(
        thermal_evidence=0.21,
        radar_evidence=0.92,
        audio_evidence=0.74,
        occlusion=0.90,
        noise=0.30,
    )

    print_result(
        "SCENARIO 2 — HEAVY RUBBLE",
        rubble,
    )

    # Scenario 3:
    # Hot machinery / thermal false positive.
    hot_object = fuse_sensors(
        thermal_evidence=0.91,
        radar_evidence=0.12,
        audio_evidence=0.08,
        occlusion=0.20,
        noise=0.20,
    )

    print_result(
        "SCENARIO 3 — HOT OBJECT",
        hot_object,
    )

    print("\n" + "=" * 60)
    print("SENSE-X FUSION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()