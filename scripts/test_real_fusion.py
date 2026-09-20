import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from datasets import load_dataset

from src.localization.human_localizer import get_human_localizer
from src.localization.radar_range import get_radar_range_lookup
from src.sensors.thermal_model import get_thermal_model
from src.sensors.radar_model import get_radar_model
from src.sensors.audio_model import get_audio_model
from src.fusion.oasf_engine import get_fusion_engine
from src.mission.priority_engine import get_priority_engine
from src.mission.active_sensing import get_active_sensing_controller

def main():
    print("=" * 70)
    print("SENSE-X — REAL 3-SENSOR FUSION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. THERMAL
    # ---------------------------------------------------------
    print("\n[1/7] Running thermal inference...")

    thermal_image_path = (
        "data/processed/flir_yolo/images/thermal/val/"
        "pair_00592_frame_002040.png"
    )

    thermal_model = get_thermal_model()

    thermal_image = np.asarray(
        Image.open(thermal_image_path).convert("RGB")
    )

    thermal_result = thermal_model.predict(thermal_image)

    thermal_evidence = thermal_result["human_probability"]

    # Get thermal bounding boxes
    thermal_boxes = thermal_result["boxes"]

    best_thermal_box = None

    if thermal_boxes:
        best_thermal_box = max(
            thermal_boxes,
            key=lambda box: box["confidence"],
        )

    print(f"Thermal human evidence: {thermal_evidence:.4f}")
    print(
        f"Thermal detections:      "
        f"{thermal_result['detections']}"
    )

    if best_thermal_box is not None:
        print(
            f"Best thermal confidence: "
            f"{best_thermal_box['confidence']:.4f}"
        )
        print(
            f"Best thermal bbox:        "
            f"{best_thermal_box['bbox']}"
        )

    # ---------------------------------------------------------
    # 2. RADAR
    # ---------------------------------------------------------
    print("\n[2/7] Running radar inference...")

    radar_path = Path("data/processed/radar")

    X_test = np.load(radar_path / "X_test.npy")
    y_test = np.load(radar_path / "y_test.npy")

    radar_model = get_radar_model()

    radar_input = X_test[0]
    radar_label = int(y_test[0])

    radar_result = radar_model.predict(radar_input)

    radar_evidence = radar_result["human_probability"]

    print(f"Radar human evidence:   {radar_evidence:.4f}")
    print(f"Radar actual label:      {radar_label}")
    print(
        f"Radar prediction:        "
        f"{radar_result['label']}"
    )

    # ---------------------------------------------------------
    # 3. AUDIO
    # ---------------------------------------------------------
    print("\n[3/7] Running audio inference...")

    audio_ds = load_dataset(
        "ahlab-drone-project/DroneAudioSet",
        "drone-with-source",
        split="train_001",
        streaming=True,
    )

    audio_sample = next(iter(audio_ds))

    audio = audio_sample["audio"]

    audio_array = np.asarray(
        audio["array"],
        dtype=np.float32,
    )

    sampling_rate = audio["sampling_rate"]

    audio_model = get_audio_model()

    audio_result = audio_model.predict(
        audio_array,
        sample_rate=sampling_rate,
        number_of_clips=3,
    )

    # IMPORTANT:
    # This is source-present evidence,
    # not calibrated human probability.
    audio_evidence = audio_result[
        "source_present_probability"
    ]

    print(
        f"Audio source evidence:  "
        f"{audio_evidence:.4f}"
    )

    print(
        f"Audio prediction:        "
        f"{audio_result['label']}"
    )

    print(
        f"Audio clips analyzed:    "
        f"{audio_result['clips_analyzed']}"
    )

    # ---------------------------------------------------------
    # 4. ADAPTIVE FUSION
    # ---------------------------------------------------------
    print("\n[4/7] Running adaptive sensor fusion...")

    fusion_engine = get_fusion_engine()

    fusion_result = fusion_engine.fuse(
        thermal_evidence=thermal_evidence,
        radar_evidence=radar_evidence,
        audio_evidence=audio_evidence,
        occlusion=0.50,
        noise=0.30,
    )

    print("\n" + "-" * 70)
    print("REAL 3-SENSOR FUSION RESULT")
    print("-" * 70)

    print(
        f"Thermal evidence:       "
        f"{fusion_result.thermal_evidence:.4f}"
    )

    print(
        f"Radar evidence:         "
        f"{fusion_result.radar_evidence:.4f}"
    )

    print(
        f"Audio evidence:         "
        f"{fusion_result.audio_evidence:.4f}"
    )

    print("\nAdaptive weights:")

    print(
        f"  Thermal:              "
        f"{fusion_result.thermal_weight:.4f}"
    )

    print(
        f"  Radar:                "
        f"{fusion_result.radar_weight:.4f}"
    )

    print(
        f"  Audio:                "
        f"{fusion_result.audio_weight:.4f}"
    )

    print(
        f"\nFusion score:            "
        f"{fusion_result.human_score:.4f}"
    )

    print(
        f"Uncertainty:             "
        f"{fusion_result.uncertainty:.4f}"
    )

    print(
        f"Decision:                "
        f"{fusion_result.decision}"
    )

    print("\nExplanation:")
    print(f"  {fusion_result.explanation}")

    # ---------------------------------------------------------
    # 5. MISSION PRIORITY
    # ---------------------------------------------------------
    print("\n[5/7] Calculating mission priority...")

    priority_engine = get_priority_engine()

    priority_result = priority_engine.calculate_priority(
        human_probability=fusion_result.human_score,
        urgency=0.80,
        distance_factor=0.70,
        obstruction=0.50,
    )

    print("\n" + "-" * 70)
    print("MISSION PRIORITY RESULT")
    print("-" * 70)

    print(
        f"Priority score:         "
        f"{priority_result.priority_score:.4f}"
    )

    print(
        f"Priority level:         "
        f"{priority_result.priority_level}"
    )

    print(
        f"Recommended action:     "
        f"{priority_result.recommended_action}"
    )

    # ---------------------------------------------------------
       # ---------------------------------------------------------
    # 6. HUMAN LOCALIZATION
    # ---------------------------------------------------------
    print("\n[6/7] Localizing detected human...")

    localizer = get_human_localizer()

    # Get recorded radar range from the UniWA dataset.
    # This is dataset metadata, not a model-estimated range.
    radar_range_lookup = get_radar_range_lookup()
    radar_range_m = radar_range_lookup.get_first_human_distance()

    localization_result = localizer.localize(
        bbox=(
            best_thermal_box["bbox"]
            if best_thermal_box is not None
            else None
        ),
        image_width=thermal_image.shape[1],
        image_height=thermal_image.shape[0],
        fusion_confidence=fusion_result.human_score,
        radar_range_m=radar_range_m,
    )

    print("\n" + "-" * 70)
    print("HUMAN LOCALIZATION RESULT")
    print("-" * 70)

    print(
        f"Relative X (forward):   "
        f"{localization_result.x:.2f} m"
    )

    print(
        f"Relative Y (lateral):    "
        f"{localization_result.y:.2f} m"
    )

    print(
        f"Relative Z (vertical):   "
        f"{localization_result.z:.2f} m"
    )

    print(
        f"Bearing:                 "
        f"{localization_result.bearing_deg:.2f}°"
    )

    print(
        f"Radar dataset range:     "
        f"{localization_result.range_m:.2f} m"
    )

    print(
        f"Localization confidence: "
        f"{localization_result.confidence:.4f}"
    )

    print(
        f"Localization source:     "
        f"{localization_result.source}"
    )
        # ---------------------------------------------------------
    # 7. ACTIVE SENSING / MISSION CONTROL
    # ---------------------------------------------------------
    print("\n[7/7] Running active sensing decision...")

    active_controller = get_active_sensing_controller()

    active_result = active_controller.decide(
        fusion_score=fusion_result.human_score,
        uncertainty=fusion_result.uncertainty,
        priority_level=priority_result.priority_level,
        localization_confidence=localization_result.confidence,
    )

    print("\n" + "-" * 70)
    print("ACTIVE SENSING DECISION")
    print("-" * 70)

    print(
        f"Action:                  "
        f"{active_result.action}"
    )

    print(
        f"Reason:                  "
        f"{active_result.reason}"
    )

    print(
        f"Next step:               "
        f"{active_result.next_step}"
    )

    # ---------------------------------------------------------
    # PIPELINE COMPLETE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("REAL SENSOR-TO-MISSION PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()