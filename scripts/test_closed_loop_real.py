import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
from PIL import Image
from datasets import load_dataset

from src.sensors.thermal_model import get_thermal_model
from src.sensors.radar_model import get_radar_model
from src.sensors.audio_model import get_audio_model
from src.fusion.oasf_engine import get_fusion_engine
from src.localization.human_localizer import get_human_localizer
from src.localization.radar_range import get_radar_range_lookup
from src.mission.priority_engine import get_priority_engine
from src.mission.active_sensing import get_active_sensing_controller
from src.mission.uav_controller import (
    get_uav_controller,
    UAVPosition,
)


# ============================================================
# CONFIGURATION
# ============================================================

THERMAL_IMAGE_1 = (
    "data/processed/flir_yolo/images/thermal/val/"
    "pair_00016_frame_002857.png"
)

THERMAL_IMAGE_2 = (
    "data/processed/flir_yolo/images/thermal/val/"
    "pair_00002_frame_006156.png"
)

RADAR_X_TEST = "data/processed/radar/X_test.npy"
RADAR_Y_TEST = "data/processed/radar/y_test.npy"

# Observation 1:
# Radar index 0 = real human sample
# Observation 2:
# Radar index 1 = real human sample
RADAR_INDEX_1 = 0
RADAR_INDEX_2 = 1

# Audio configuration
AUDIO_CONFIG = "drone-only"
AUDIO_SPLIT = "train_001"

# Observation 1:
# 5th streamed sample previously produced weak
# source-present evidence (~0.3439).
AUDIO_INDEX_1 = 4

# Observation 2:
# 1st streamed sample previously produced
# source-present evidence (~0.4501).
AUDIO_INDEX_2 = 0

# False means the real active-sensing controller
# must request the rescan.
FORCE_RESCAN_DEMO = False


# ============================================================
# HELPERS
# ============================================================

def print_separator():
    print("\n" + "=" * 70)


def load_audio_samples(number_of_samples=5):
    """
    Stream real DroneAudioSet samples.

    The audio dataset is independent from the FLIR and
    UniWA radar datasets. These samples are therefore used
    for real sensor-model integration testing, not as
    temporally synchronized measurements of the same scene.
    """

    print("Loading real audio observations...")

    dataset = load_dataset(
        "ahlab-drone-project/DroneAudioSet",
        AUDIO_CONFIG,
        split=AUDIO_SPLIT,
        streaming=True,
    )

    iterator = iter(dataset)
    samples = []

    for i in range(number_of_samples):
        try:
            sample = next(iterator)
            samples.append(sample)

            file_path = sample.get(
                "file_path",
                "unknown",
            )

            print(
                f"Audio sample {i + 1}: "
                f"{file_path}"
            )

        except StopIteration:
            print(
                f"WARNING: Could only load "
                f"{len(samples)} audio samples."
            )
            break

    return samples


def prepare_audio_observation(
    audio_sample,
    audio_model,
):
    """
    Run the real audio model on one streamed
    DroneAudioSet sample.
    """

    audio = audio_sample["audio"]

    audio_array = np.asarray(
        audio["array"],
        dtype=np.float32,
    )

    sampling_rate = audio["sampling_rate"]

    result = audio_model.predict(
        audio_array,
        sample_rate=sampling_rate,
        number_of_clips=3,
    )

    return result


def load_radar_data():
    """
    Load the prepared radar test split.
    """

    X_test = np.load(RADAR_X_TEST)
    y_test = np.load(RADAR_Y_TEST)

    return X_test, y_test


def load_thermal_image(
    thermal_image_path,
):
    """
    Load a thermal PNG as a NumPy array.

    ThermalModel.predict() expects a NumPy image,
    not a file path.
    """

    thermal_path = Path(thermal_image_path)

    if not thermal_path.exists():
        raise FileNotFoundError(
            f"Thermal image not found: {thermal_path}"
        )

    image = Image.open(thermal_path)

    image_array = np.asarray(image)

    return image_array


def run_observation(
    observation_number,
    thermal_image_path,
    radar_input,
    radar_actual_label,
    audio_result,
    thermal_model,
    radar_model,
    fusion_engine,
    localizer,
    radar_range_lookup,
    priority_engine,
    active_controller,
):
    """
    Run one complete real-data multimodal observation.
    """

    print_separator()

    print(
        f"OBSERVATION {observation_number}"
    )

    print_separator()

    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    thermal_path = Path(
        thermal_image_path
    )

    thermal_image = load_thermal_image(
        thermal_path
    )

    thermal_result = thermal_model.predict(
        thermal_image
    )

    thermal_evidence = float(
        thermal_result["human_probability"]
    )

    print(
        f"Thermal image: "
        f"{thermal_path.name}"
    )

    print(
        f"Thermal image shape: "
        f"{thermal_image.shape}"
    )

    print(
        f"Thermal evidence: "
        f"{thermal_evidence:.4f}"
    )

    print(
        f"Thermal detections: "
        f"{thermal_result['detections']}"
    )

    if thermal_result["boxes"]:

        best_bbox = max(
            thermal_result["boxes"],
            key=lambda box: box["confidence"],
        )

        bbox = best_bbox["bbox"]

        print(
            f"Best thermal bbox: "
            f"{bbox}"
        )

        print(
            f"Best thermal confidence: "
            f"{best_bbox['confidence']:.4f}"
        )

    else:

        bbox = None

        print(
            "Best thermal bbox: None"
        )

    # --------------------------------------------------------
    # RADAR
    # --------------------------------------------------------

    radar_result = radar_model.predict(
        radar_input
    )

    radar_evidence = float(
        radar_result["human_probability"]
    )

    print()
    print("Radar:")

    print(
        f"Radar human evidence: "
        f"{radar_evidence:.4f}"
    )

    print(
        f"Radar actual label: "
        f"{int(radar_actual_label)}"
    )

    print(
        f"Radar prediction: "
        f"{radar_result['label']}"
    )

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    audio_evidence = float(
        audio_result[
            "source_present_probability"
        ]
    )

    print()
    print("Audio:")

    print(
        f"Audio source evidence: "
        f"{audio_evidence:.4f}"
    )

    print(
        f"Audio prediction: "
        f"{audio_result['label']}"
    )

    print(
        f"Audio clips analyzed: "
        f"{audio_result['clips_analyzed']}"
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    fusion_result = fusion_engine.fuse(
        thermal_evidence=thermal_evidence,
        radar_evidence=radar_evidence,
        audio_evidence=audio_evidence,
        occlusion=0.50,
        noise=0.30,
    )

    print()
    print("Adaptive fusion:")

    print(
        f"Thermal {thermal_evidence:.4f}, "
        f"Radar {radar_evidence:.4f}, "
        f"Audio {audio_evidence:.4f}"
    )

    print(
        f"Fusion score: "
        f"{fusion_result.human_score:.4f}"
    )

    print(
        f"Uncertainty: "
        f"{fusion_result.uncertainty:.4f}"
    )

    print(
        f"Decision: "
        f"{fusion_result.decision}"
    )

    print(
        f"Weights: "
        f"Thermal "
        f"{fusion_result.thermal_weight:.2f}, "
        f"Radar "
        f"{fusion_result.radar_weight:.2f}, "
        f"Audio "
        f"{fusion_result.audio_weight:.2f}"
    )

    print(
        f"Explanation: "
        f"{fusion_result.explanation}"
    )

    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    localization_result = None

    if bbox is not None:

        # IMPORTANT:
        # This is the dataset-recorded radar distance.
        # It is NOT estimated from the radar signal.
        radar_range = (
            radar_range_lookup
            .get_first_human_distance()
        )

        image_height = thermal_image.shape[0]
        image_width = thermal_image.shape[1]

        localization_result = (
            localizer.localize(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                fusion_confidence=(
                    fusion_result.human_score
                ),
                radar_range_m=radar_range,
            )
        )

        print()
        print("Mission localization:")

        print(
            f"Relative X: "
            f"{localization_result.x:.2f} m"
        )

        print(
            f"Relative Y: "
            f"{localization_result.y:.2f} m"
        )

        print(
            f"Relative Z: "
            f"{localization_result.z:.2f} m"
        )

        print(
            f"Bearing: "
            f"{localization_result.bearing_deg:.2f}°"
        )

        if localization_result.range_m is not None:

            print(
                f"Dataset radar range: "
                f"{localization_result.range_m:.2f} m"
            )

        else:

            print(
                "Dataset radar range: "
                "unavailable"
            )

        print(
            f"Localization confidence: "
            f"{localization_result.confidence:.4f}"
        )

    else:

        localization_confidence = 0.0

        print()
        print(
            "Mission localization: "
            "No thermal detection available."
        )

    if localization_result is not None:

        localization_confidence = (
            localization_result.confidence
        )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority_result = (
        priority_engine.calculate_priority(
            human_probability=(
                fusion_result.human_score
            ),
            urgency=0.80,
            distance_factor=0.70,
            obstruction=0.50,
        )
    )

    print()
    print("Priority:")

    print(
        f"Priority score: "
        f"{priority_result.priority_score:.4f}"
    )

    print(
        f"Priority level: "
        f"{priority_result.priority_level}"
    )

    print(
        f"Recommended action: "
        f"{priority_result.recommended_action}"
    )

    # --------------------------------------------------------
    # ACTIVE SENSING
    # --------------------------------------------------------

    active_result = (
        active_controller.decide(
            fusion_score=(
                fusion_result.human_score
            ),
            uncertainty=(
                fusion_result.uncertainty
            ),
            priority_level=(
                priority_result.priority_level
            ),
            localization_confidence=(
                localization_confidence
            ),
        )
    )

    print()
    print("Active sensing:")

    print(
        f"Action: "
        f"{active_result.action}"
    )

    print(
        f"Reason: "
        f"{active_result.reason}"
    )

    print(
        f"Next step: "
        f"{active_result.next_step}"
    )

    return {
        "thermal": thermal_result,
        "radar": radar_result,
        "audio": audio_result,
        "fusion": fusion_result,
        "localization": localization_result,
        "priority": priority_result,
        "active": active_result,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)
    print(
        "SENSE-X — REAL SEQUENTIAL "
        "MULTIMODAL TEST"
    )
    print("=" * 70)

    print()

    print(
        "Thermal, radar and audio samples "
        "are real dataset samples."
    )

    print(
        "The datasets are independent and are "
        "NOT temporally synchronized measurements "
        "of the same physical scene."
    )

    print(
        "This test validates the multimodal fusion "
        "and active-sensing software pipeline."
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print()
    print("Loading models...")

    thermal_model = (
        get_thermal_model()
    )

    radar_model = (
        get_radar_model()
    )

    audio_model = (
        get_audio_model()
    )

    fusion_engine = (
        get_fusion_engine()
    )

    localizer = (
        get_human_localizer()
    )

    radar_range_lookup = (
        get_radar_range_lookup()
    )

    priority_engine = (
        get_priority_engine()
    )

    active_controller = (
        get_active_sensing_controller()
    )

    uav_controller = (
        get_uav_controller()
    )

    # --------------------------------------------------------
    # LOAD RADAR
    # --------------------------------------------------------

    print(
        "Loading radar test data..."
    )

    X_test, y_test = load_radar_data()

    if RADAR_INDEX_1 >= len(X_test):

        raise IndexError(
            f"RADAR_INDEX_1={RADAR_INDEX_1} "
            f"but radar test set has "
            f"{len(X_test)} samples."
        )

    if RADAR_INDEX_2 >= len(X_test):

        raise IndexError(
            f"RADAR_INDEX_2={RADAR_INDEX_2} "
            f"but radar test set has "
            f"{len(X_test)} samples."
        )

    radar_input_1 = (
        X_test[RADAR_INDEX_1]
    )

    radar_label_1 = (
        y_test[RADAR_INDEX_1]
    )

    radar_input_2 = (
        X_test[RADAR_INDEX_2]
    )

    radar_label_2 = (
        y_test[RADAR_INDEX_2]
    )

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    audio_samples = load_audio_samples(
        number_of_samples=5
    )

    if len(audio_samples) <= AUDIO_INDEX_1:

        raise RuntimeError(
            "Not enough audio samples were "
            "loaded for Observation 1."
        )

    if len(audio_samples) <= AUDIO_INDEX_2:

        raise RuntimeError(
            "Not enough audio samples were "
            "loaded for Observation 2."
        )

    audio_result_1 = (
        prepare_audio_observation(
            audio_samples[AUDIO_INDEX_1],
            audio_model,
        )
    )

    audio_result_2 = (
        prepare_audio_observation(
            audio_samples[AUDIO_INDEX_2],
            audio_model,
        )
    )

    # --------------------------------------------------------
    # UAV INITIAL POSITION
    # --------------------------------------------------------

    uav_position = UAVPosition(
        x=0.0,
        y=0.0,
        z=10.0,
    )

    print()

    print(
        f"Initial UAV position: "
        f"({uav_position.x:.2f}, "
        f"{uav_position.y:.2f}, "
        f"{uav_position.z:.2f})"
    )

    # ========================================================
    # OBSERVATION 1
    # ========================================================

    observation_1 = run_observation(
        observation_number=1,
        thermal_image_path=THERMAL_IMAGE_1,
        radar_input=radar_input_1,
        radar_actual_label=radar_label_1,
        audio_result=audio_result_1,
        thermal_model=thermal_model,
        radar_model=radar_model,
        fusion_engine=fusion_engine,
        localizer=localizer,
        radar_range_lookup=radar_range_lookup,
        priority_engine=priority_engine,
        active_controller=active_controller,
    )

    active_action_1 = (
        observation_1["active"].action
    )

    print_separator()

    print(
        "ACTIVE SENSING RESULT — "
        "OBSERVATION 1"
    )

    print_separator()

    print(
        f"Controller action: "
        f"{active_action_1}"
    )

    # --------------------------------------------------------
    # FORCE OPTION
    # --------------------------------------------------------

    if FORCE_RESCAN_DEMO:

        print()

        print(
            "WARNING: "
            "FORCE_RESCAN_DEMO=True"
        )

        print(
            "The rescan is being forced "
            "for demonstration purposes."
        )

        active_action_1 = (
            "REPOSITION_AND_RESCAN"
        )

    # ========================================================
    # DECISION: RESCAN OR INVESTIGATE
    # ========================================================

    if active_action_1 == (
        "REPOSITION_AND_RESCAN"
    ):

        print()

        print(
            "Controller requested "
            "another observation."
        )

        # ----------------------------------------------------
        # UAV REPOSITION
        # ----------------------------------------------------

        uav_action = (
            uav_controller.reposition(
                position=uav_position,
                direction="lateral",
            )
        )

        uav_position = (
            uav_action.new_position
        )

        print()

        print("UAV reposition:")

        print(
            f"Action: "
            f"{uav_action.action}"
        )

        print(
            f"Old position: "
            f"({uav_action.old_position.x:.2f}, "
            f"{uav_action.old_position.y:.2f}, "
            f"{uav_action.old_position.z:.2f})"
        )

        print(
            f"New position: "
            f"({uav_action.new_position.x:.2f}, "
            f"{uav_action.new_position.y:.2f}, "
            f"{uav_action.new_position.z:.2f})"
        )

        print(
            f"Movement distance: "
            f"{uav_action.movement_distance:.2f} m"
        )

        # ====================================================
        # OBSERVATION 2
        # ====================================================

        observation_2 = run_observation(
            observation_number=2,
            thermal_image_path=THERMAL_IMAGE_2,
            radar_input=radar_input_2,
            radar_actual_label=radar_label_2,
            audio_result=audio_result_2,
            thermal_model=thermal_model,
            radar_model=radar_model,
            fusion_engine=fusion_engine,
            localizer=localizer,
            radar_range_lookup=radar_range_lookup,
            priority_engine=priority_engine,
            active_controller=active_controller,
        )

        active_action_2 = (
            observation_2["active"].action
        )

        print_separator()

        print(
            "FINAL CLOSED-LOOP RESULT"
        )

        print_separator()

        print(
            "Observation 1: "
            f"{observation_1['fusion'].decision}"
        )

        print(
            "Observation 1 action: "
            f"{active_action_1}"
        )

        print(
            "UAV repositioned: "
            f"{uav_action.movement_distance:.2f} m"
        )

        print(
            "Observation 2: "
            f"{observation_2['fusion'].decision}"
        )

        print(
            "Observation 2 action: "
            f"{active_action_2}"
        )

        print(
            f"Final UAV position: "
            f"({uav_position.x:.2f}, "
            f"{uav_position.y:.2f}, "
            f"{uav_position.z:.2f})"
        )

    else:

        print()

        print(
            "The real active-sensing controller "
            "did not request a rescan."
        )

        print(
            "No second observation was "
            "artificially requested."
        )

        print_separator()

        print(
            "FINAL RESULT"
        )

        print_separator()

        print(
            f"Final action: "
            f"{active_action_1}"
        )

        print(
            f"Final UAV position: "
            f"({uav_position.x:.2f}, "
            f"{uav_position.y:.2f}, "
            f"{uav_position.z:.2f})"
        )

    print()

    print("=" * 70)

    print(
        "SENSE-X REAL SEQUENTIAL TEST COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()