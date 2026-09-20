from pathlib import Path

import cv2
import numpy as np
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# REAL DATASET OBSERVATIONS
# ============================================================

THERMAL_IMAGE_1 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flir_yolo"
    / "images"
    / "thermal"
    / "val"
    / "pair_00016_frame_002857.png"
)

THERMAL_IMAGE_2 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flir_yolo"
    / "images"
    / "thermal"
    / "val"
    / "pair_00002_frame_006156.png"
)

RADAR_X_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "radar"
    / "X_test.npy"
)

RADAR_Y_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "radar"
    / "y_test.npy"
)

RADAR_INDEX_1 = 0
RADAR_INDEX_2 = 1

AUDIO_CONFIG = "drone-only"
AUDIO_SPLIT = "train_001"

AUDIO_INDEX_1 = 4
AUDIO_INDEX_2 = 0


# ============================================================
# SERVICE
# ============================================================

class MissionService:

    def __init__(self):

        self.models_loaded = False
        self.audio_loaded = False

        self.thermal_model = None
        self.radar_model = None
        self.audio_model = None

        self.fusion_engine = None
        self.localizer = None
        self.radar_range_lookup = None

        self.priority_engine = None
        self.active_controller = None
        self.uav_controller = None

        self.X_test = None
        self.y_test = None

        self.audio_samples = []

        self.uav_position = UAVPosition(
            x=0.0,
            y=0.0,
            z=10.0,
        )

        self.observation_number = 0

        self.last_observation = None

        self.mission_history = []

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    def load_models(self):

        if self.models_loaded:
            return

        print("=" * 70)
        print("SENSE-X BACKEND INITIALIZATION")
        print("=" * 70)

        print("Loading thermal model...")
        self.thermal_model = get_thermal_model()

        print("Loading radar model...")
        self.radar_model = get_radar_model()

        print("Loading audio model...")
        self.audio_model = get_audio_model()

        print("Loading fusion engine...")
        self.fusion_engine = get_fusion_engine()

        print("Loading localization engine...")
        self.localizer = get_human_localizer()

        self.radar_range_lookup = (
            get_radar_range_lookup()
        )

        print("Loading mission engines...")
        self.priority_engine = get_priority_engine()

        self.active_controller = (
            get_active_sensing_controller()
        )

        self.uav_controller = get_uav_controller()

        print("Loading radar test data...")

        self.X_test = np.load(
            RADAR_X_TEST
        )

        self.y_test = np.load(
            RADAR_Y_TEST
        )

        self.models_loaded = True

        print("SENSE-X backend ready.")
        print("=" * 70)

    # ========================================================
    # AUDIO
    # ========================================================

    def load_audio_samples(self):

        if self.audio_loaded:
            return

        self.load_models()

        print("Loading DroneAudioSet stream...")

        dataset = load_dataset(
            "ahlab-drone-project/DroneAudioSet",
            AUDIO_CONFIG,
            split=AUDIO_SPLIT,
            streaming=True,
        )

        iterator = iter(dataset)

        samples = []

        for _ in range(5):
            try:
                samples.append(
                    next(iterator)
                )
            except StopIteration:
                break

        if len(samples) < 5:
            raise RuntimeError(
                "Unable to load the required "
                "DroneAudioSet samples."
            )

        self.audio_samples = samples

        self.audio_loaded = True

        print(
            f"Loaded {len(samples)} "
            "real audio samples."
        )

    # ========================================================
    # THERMAL
    # ========================================================

    def run_thermal(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Thermal image not found: "
                f"{image_path}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV failed to load "
                f"{image_path}"
            )

        result = self.thermal_model.predict(
            image
        )

        boxes = result.get(
            "boxes",
            [],
        )

        best_box = None

        if boxes:
            best_box = max(
                boxes,
                key=lambda item: item["confidence"],
            )

        return {
            "image_path": str(image_path),
            "image_name": image_path.name,

            "width": int(image.shape[1]),
            "height": int(image.shape[0]),

            "evidence": float(
                result["human_probability"]
            ),

            "detections": int(
                result["detections"]
            ),

            "bbox": (
                best_box["bbox"]
                if best_box is not None
                else None
            ),

            "best_confidence": (
                float(best_box["confidence"])
                if best_box is not None
                else None
            ),
        }

    # ========================================================
    # RADAR
    # ========================================================

    def run_radar(self, index):

        radar_input = self.X_test[index]

        actual_label = int(
            self.y_test[index]
        )

        result = self.radar_model.predict(
            radar_input
        )

        return {
            "index": int(index),

            "evidence": float(
                result["human_probability"]
            ),

            "actual_label": actual_label,

            "prediction": result["label"],
        }

    # ========================================================
    # AUDIO
    # ========================================================

    def run_audio(self, index):

        self.load_audio_samples()

        sample = self.audio_samples[index]

        audio = sample["audio"]

        audio_array = np.asarray(
            audio["array"],
            dtype=np.float32,
        )

        sampling_rate = int(
            audio["sampling_rate"]
        )

        result = self.audio_model.predict(
            audio_array,
            sample_rate=sampling_rate,
            number_of_clips=3,
        )

        return {
            "index": int(index),

            "file_path": sample.get(
                "file_path",
                "unknown",
            ),

            "sampling_rate": sampling_rate,

            "evidence": float(
                result[
                    "source_present_probability"
                ]
            ),

            "drone_only_probability": float(
                result[
                    "drone_only_probability"
                ]
            ),

            "prediction": result["label"],

            "clips_analyzed": int(
                result["clips_analyzed"]
            ),
        }

    # ========================================================
    # OBSERVATION
    # ========================================================

    def run_observation(
        self,
        observation_number,
    ):

        self.load_models()

        if observation_number == 1:

            thermal_path = THERMAL_IMAGE_1
            radar_index = RADAR_INDEX_1
            audio_index = AUDIO_INDEX_1

        elif observation_number == 2:

            thermal_path = THERMAL_IMAGE_2
            radar_index = RADAR_INDEX_2
            audio_index = AUDIO_INDEX_2

        else:

            raise ValueError(
                "Only observations 1 and 2 "
                "are currently configured."
            )

        # ----------------------------------------------------
        # SENSOR INFERENCE
        # ----------------------------------------------------

        thermal = self.run_thermal(
            thermal_path
        )

        radar = self.run_radar(
            radar_index
        )

        audio = self.run_audio(
            audio_index
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion = self.fusion_engine.fuse(
            thermal_evidence=thermal["evidence"],
            radar_evidence=radar["evidence"],
            audio_evidence=audio["evidence"],
            occlusion=0.50,
            noise=0.30,
        )

        # ----------------------------------------------------
        # LOCALIZATION
        # ----------------------------------------------------

        localization_data = None
        localization_confidence = 0.0

        if thermal["bbox"] is not None:

            radar_range = (
                self.radar_range_lookup
                .get_first_human_distance()
            )

            localization = self.localizer.localize(
                bbox=thermal["bbox"],
                image_width=thermal["width"],
                image_height=thermal["height"],
                fusion_confidence=(
                    fusion.human_score
                ),
                radar_range_m=radar_range,
            )

            localization_confidence = float(
                localization.confidence
            )

            localization_data = {
                "x": float(localization.x),
                "y": float(localization.y),
                "z": float(localization.z),

                "bearing_deg": float(
                    localization.bearing_deg
                ),

                "range_m": (
                    float(localization.range_m)
                    if localization.range_m
                    is not None
                    else None
                ),

                "confidence": float(
                    localization.confidence
                ),

                "source": localization.source,
            }

        # ----------------------------------------------------
        # PRIORITY
        # ----------------------------------------------------

        priority = (
            self.priority_engine
            .calculate_priority(
                human_probability=(
                    fusion.human_score
                ),
                urgency=0.80,
                distance_factor=0.70,
                obstruction=0.50,
            )
        )

        # ----------------------------------------------------
        # ACTIVE SENSING
        # ----------------------------------------------------

        active = self.active_controller.decide(
            fusion_score=fusion.human_score,
            uncertainty=fusion.uncertainty,
            priority_level=priority.priority_level,
            localization_confidence=(
                localization_confidence
            ),
        )

        # ----------------------------------------------------
        # SERIALIZABLE RESPONSE
        # ----------------------------------------------------

        observation = {

            "observation": observation_number,

            "thermal": thermal,

            "radar": radar,

            "audio": audio,

            "fusion": {
                "score": float(
                    fusion.human_score
                ),

                "uncertainty": float(
                    fusion.uncertainty
                ),

                "decision": fusion.decision,

                "weights": {
                    "thermal": float(
                        fusion.thermal_weight
                    ),

                    "radar": float(
                        fusion.radar_weight
                    ),

                    "audio": float(
                        fusion.audio_weight
                    ),
                },

                "explanation": (
                    fusion.explanation
                ),
            },

            "localization": localization_data,

            "priority": {
                "score": float(
                    priority.priority_score
                ),

                "level": (
                    priority.priority_level
                ),

                "recommended_action": (
                    priority.recommended_action
                ),
            },

            "active_sensing": {
                "action": active.action,
                "reason": active.reason,
                "next_step": active.next_step,
            },

            "uav": {
                "x": float(
                    self.uav_position.x
                ),

                "y": float(
                    self.uav_position.y
                ),

                "z": float(
                    self.uav_position.z
                ),
            },
        }

        self.observation_number = (
            observation_number
        )

        self.last_observation = observation

        self.mission_history.append(
            observation
        )

        return observation

    # ========================================================
    # UAV MOVEMENT
    # ========================================================

    def reposition_uav(self):

        movement = (
            self.uav_controller.reposition(
                self.uav_position,
                direction="lateral",
            )
        )

        self.uav_position = (
            movement.new_position
        )

        return {
            "action": movement.action,

            "movement_distance": float(
                movement.movement_distance
            ),

            "old_position": {
                "x": float(
                    movement.old_position.x
                ),
                "y": float(
                    movement.old_position.y
                ),
                "z": float(
                    movement.old_position.z
                ),
            },

            "new_position": {
                "x": float(
                    movement.new_position.x
                ),
                "y": float(
                    movement.new_position.y
                ),
                "z": float(
                    movement.new_position.z
                ),
            },
        }

    # ========================================================
    # COMPLETE CLOSED LOOP
    # ========================================================

    def run_closed_loop(self):

        self.reset()

        observation_1 = self.run_observation(
            1
        )

        result = {
            "observation_1": observation_1,
            "movement": None,
            "observation_2": None,
        }

        action = (
            observation_1[
                "active_sensing"
            ]["action"]
        )

        if action == "REPOSITION_AND_RESCAN":

            movement = self.reposition_uav()

            observation_2 = (
                self.run_observation(2)
            )

            result["movement"] = movement
            result["observation_2"] = (
                observation_2
            )

        return result

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        return {
            "system": "SENSE-X",

            "backend": "ONLINE",

            "models_loaded": (
                self.models_loaded
            ),

            "audio_loaded": (
                self.audio_loaded
            ),

            "observation": (
                self.observation_number
            ),

            "uav": {
                "x": float(
                    self.uav_position.x
                ),

                "y": float(
                    self.uav_position.y
                ),

                "z": float(
                    self.uav_position.z
                ),
            },

            "last_observation": (
                self.last_observation
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.uav_position = UAVPosition(
            x=0.0,
            y=0.0,
            z=10.0,
        )

        self.observation_number = 0

        self.last_observation = None

        self.mission_history = []

        return {
            "status": "RESET",
            "message": (
                "SENSE-X mission state reset."
            ),
        }


# ============================================================
# SINGLETON
# ============================================================

_mission_service = None


def get_mission_service():

    global _mission_service

    if _mission_service is None:
        _mission_service = MissionService()

    return _mission_service