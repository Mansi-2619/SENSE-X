import sys
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

INDEX_FILE = DASHBOARD_DIR / "index.html"
STYLE_FILE = DASHBOARD_DIR / "style.css"
JS_FILE = DASHBOARD_DIR / "app.js"


THERMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flir_yolo"
    / "images"
    / "thermal"
    / "val"
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
RADAR_DISTANCE_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "radar"
    / "distance_test.npy"
)


AUDIO_CONFIG = "drone-only"
AUDIO_SPLIT = "train_001"

# Number of real DroneAudioSet samples cached for dynamic selection.
AUDIO_POOL_SIZE = 20


# ============================================================
# DYNAMIC DATASET SELECTION STATE
# ============================================================

_last_thermal_paths = set()
_last_radar_indices = set()
_last_audio_indices = set()

# Previous tests:
# index 4 -> ~0.3439
# index 0 -> ~0.4501
AUDIO_INDEX_1 = 4
AUDIO_INDEX_2 = 0


# ============================================================
# VERIFY REQUIRED FILES
# ============================================================

for required_file in (
    INDEX_FILE,
    STYLE_FILE,
    JS_FILE,
):
    if not required_file.exists():
        raise RuntimeError(
            f"Required dashboard file missing:\n"
            f"{required_file}"
        )


if not THERMAL_DIR.exists():
    raise RuntimeError(
        f"Thermal directory does not exist:\n"
        f"{THERMAL_DIR}"
    )


# ============================================================
# SENSE-X BACKEND IMPORTS
# ============================================================

from src.sensors.thermal_model import get_thermal_model
from src.sensors.radar_model import get_radar_model
from src.sensors.audio_model import get_audio_model

from src.fusion.oasf_engine import get_fusion_engine

from src.localization.human_localizer import (
    get_human_localizer,
)

from src.localization.radar_range import (
    get_radar_range_lookup,
)

from src.mission.priority_engine import (
    get_priority_engine,
)

from src.mission.active_sensing import (
    get_active_sensing_controller,
)

from src.mission.uav_controller import (
    get_uav_controller,
    UAVPosition,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SENSE-X Rescue Command",
    version="2.0.0",
    description=(
        "SENSE-X multimodal autonomous search "
        "and rescue command backend."
    ),
)


# ============================================================
# STATIC DASHBOARD
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(DASHBOARD_DIR)),
    name="static",
)


app.mount(
    "/thermal",
    StaticFiles(directory=str(THERMAL_DIR)),
    name="thermal",
)


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(
        INDEX_FILE,
        media_type="text/html",
    )


# Compatibility routes.
# Your index.html currently uses /static/*, but these allow
# direct /style.css and /app.js requests as well.

@app.get("/style.css", include_in_schema=False)
async def dashboard_css():
    return FileResponse(
        STYLE_FILE,
        media_type="text/css",
    )


@app.get("/app.js", include_in_schema=False)
async def dashboard_js():
    return FileResponse(
        JS_FILE,
        media_type="application/javascript",
    )


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_models: dict[str, Any] = {}

_radar_data = None
_audio_samples = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load all SENSE-X models once and cache them.

    This prevents the dashboard from reloading neural-network
    models every time START MISSION is clicked.
    """

    global _models

    if _models:
        return _models

    print()
    print("=" * 70)
    print("INITIALIZING SENSE-X BACKEND")
    print("=" * 70)

    print("Loading thermal model...")
    thermal_model = get_thermal_model()

    print("Loading radar model...")
    radar_model = get_radar_model()

    print("Loading audio model...")
    audio_model = get_audio_model()

    print("Loading OASF engine...")
    fusion_engine = get_fusion_engine()

    print("Loading human localizer...")
    localizer = get_human_localizer()

    print("Loading radar range lookup...")
    radar_range_lookup = get_radar_range_lookup()

    print("Loading priority engine...")
    priority_engine = get_priority_engine()

    print("Loading active sensing controller...")
    active_controller = (
        get_active_sensing_controller()
    )

    print("Loading UAV controller...")
    uav_controller = get_uav_controller()

    _models = {
        "thermal": thermal_model,
        "radar": radar_model,
        "audio": audio_model,
        "fusion": fusion_engine,
        "localizer": localizer,
        "radar_range": radar_range_lookup,
        "priority": priority_engine,
        "active": active_controller,
        "uav": uav_controller,
    }

    print("SENSE-X backend initialized.")
    print("=" * 70)
    print()

    return _models


# ============================================================
# RADAR DATA
# ============================================================

def load_radar_data():
    """
    Load the processed radar test split.

    IMPORTANT:
    X_test[i], y_test[i], and distance_test[i]
    correspond to the same processed radar observation.

    distance_test contains:
        finite distance -> human-present sample
        NaN             -> no-human sample
    """

    global _radar_data

    if _radar_data is not None:
        return _radar_data

    if not RADAR_X_TEST.exists():
        raise FileNotFoundError(
            f"Radar X_test not found:\n"
            f"{RADAR_X_TEST}"
        )

    if not RADAR_Y_TEST.exists():
        raise FileNotFoundError(
            f"Radar y_test not found:\n"
            f"{RADAR_Y_TEST}"
        )

    if not RADAR_DISTANCE_TEST.exists():
        raise FileNotFoundError(
            f"Radar distance_test not found:\n"
            f"{RADAR_DISTANCE_TEST}"
        )

    print("Loading radar test data...")

    X_test = np.load(RADAR_X_TEST)
    y_test = np.load(RADAR_Y_TEST)
    distance_test = np.load(RADAR_DISTANCE_TEST)

    if not (
        len(X_test)
        == len(y_test)
        == len(distance_test)
    ):
        raise RuntimeError(
            "Radar processed arrays are not aligned: "
            f"X={len(X_test)}, "
            f"y={len(y_test)}, "
            f"distance={len(distance_test)}"
        )

    print(
        "Radar test data loaded:",
        f"{len(X_test)} samples"
    )

    print(
        "Radar human samples:",
        int(np.sum(y_test == 1))
    )

    print(
        "Radar non-human samples:",
        int(np.sum(y_test == 0))
    )

    _radar_data = (
        X_test,
        y_test,
        distance_test,
    )

    return _radar_data

# ============================================================
# AUDIO DATA
# ============================================================

def load_audio_samples(
    number_of_samples: int = 5,
):
    """
    Stream real DroneAudioSet samples.

    The loader attempts to collect up to number_of_samples samples.

    If the streaming split contains fewer samples than requested,
    all available samples are used instead of failing the entire
    SENSE-X mission.

    Samples are cached after the first successful load.
    """

    global _audio_samples

    # --------------------------------------------------------
    # RETURN CACHE WHEN AVAILABLE
    # --------------------------------------------------------

    if _audio_samples is not None:
        if len(_audio_samples) > 0:

            print(
                f"Using cached DroneAudioSet pool: "
                f"{len(_audio_samples)} samples."
            )

            return _audio_samples

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("Loading real DroneAudioSet samples...")

    dataset = load_dataset(
        "ahlab-drone-project/DroneAudioSet",
        AUDIO_CONFIG,
        split=AUDIO_SPLIT,
        streaming=True,
    )

    iterator = iter(dataset)

    samples = []

    # --------------------------------------------------------
    # STREAM AVAILABLE SAMPLES
    # --------------------------------------------------------

    for index in range(number_of_samples):

        try:
            sample = next(iterator)

        except StopIteration:

            print(
                f"DroneAudioSet stream ended after "
                f"{len(samples)} samples."
            )

            break

        samples.append(sample)

        print(
            f"Audio sample {index + 1}: "
            f"{sample.get('file_path', 'unknown')}"
        )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if len(samples) == 0:

        raise RuntimeError(
            "DroneAudioSet did not provide any "
            "usable streaming samples."
        )

    # Having fewer samples than requested is NOT fatal.
    if len(samples) < number_of_samples:

        print()
        print(
            f"WARNING: Requested "
            f"{number_of_samples} audio samples, "
            f"but DroneAudioSet provided "
            f"{len(samples)} usable samples."
        )

        print(
            "Continuing mission with the "
            "available real audio samples."
        )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    _audio_samples = samples

    print()
    print(
        f"Audio pool ready: "
        f"{len(_audio_samples)} real samples."
    )

    return _audio_samples


# ============================================================
# AUDIO INFERENCE
# ============================================================

def run_audio_inference(
    audio_sample,
    audio_model,
):
    audio = audio_sample["audio"]

    audio_array = np.asarray(
        audio["array"],
        dtype=np.float32,
    )

    sample_rate = int(
        audio["sampling_rate"]
    )

    result = audio_model.predict(
        audio_array,
        sample_rate=sample_rate,
        number_of_clips=3,
    )

    return result


# ============================================================
# SAFE SERIALIZATION
# ============================================================

def safe_float(value, default=0.0):
    try:
        number = float(value)

        if np.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return float(default)


def safe_int(value, default=0):
    try:
        return int(value)

    except (TypeError, ValueError):
        return int(default)

# ============================================================
# DYNAMIC REAL DATASET SAMPLE SELECTION
# ============================================================

def get_thermal_candidates():
    """
    Return all usable thermal images from the existing FLIR dataset.
    """

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    candidates = [
        path
        for path in THERMAL_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in extensions
        )
    ]

    if not candidates:
        raise RuntimeError(
            f"No thermal images found in:\n"
            f"{THERMAL_DIR}"
        )

    return candidates


def choose_thermal_samples(count=2):
    """
    Select different real thermal frames for this mission.

    Tries not to immediately reuse the previous mission's frames.
    """

    global _last_thermal_paths

    candidates = get_thermal_candidates()

    previous = {
        str(path)
        for path in _last_thermal_paths
    }

    fresh_candidates = [
        path
        for path in candidates
        if str(path) not in previous
    ]

    pool = (
        fresh_candidates
        if len(fresh_candidates) >= count
        else candidates
    )

    if len(pool) >= count:
        selected = random.sample(
            pool,
            count,
        )
    else:
        selected = [
            random.choice(pool)
            for _ in range(count)
        ]

    _last_thermal_paths = set(selected)

    return selected


def choose_radar_indices(
    dataset_size,
    count=2,
):
    """
    Select real radar test samples dynamically.
    """

    global _last_radar_indices

    if dataset_size <= 0:
        raise RuntimeError(
            "Radar test dataset is empty."
        )

    all_indices = list(
        range(dataset_size)
    )

    fresh_indices = [
        index
        for index in all_indices
        if index not in _last_radar_indices
    ]

    pool = (
        fresh_indices
        if len(fresh_indices) >= count
        else all_indices
    )

    if len(pool) >= count:
        selected = random.sample(
            pool,
            count,
        )
    else:
        selected = [
            random.choice(pool)
            for _ in range(count)
        ]

    _last_radar_indices = set(selected)

    return selected


def choose_audio_indices(
    number_of_samples,
    count=2,
):
    """
    Select different cached real DroneAudioSet samples.
    """

    global _last_audio_indices

    if number_of_samples <= 0:
        raise RuntimeError(
            "Audio sample pool is empty."
        )

    all_indices = list(
        range(number_of_samples)
    )

    fresh_indices = [
        index
        for index in all_indices
        if index not in _last_audio_indices
    ]

    pool = (
        fresh_indices
        if len(fresh_indices) >= count
        else all_indices
    )

    if len(pool) >= count:
        selected = random.sample(
            pool,
            count,
        )
    else:
        selected = [
            random.choice(pool)
            for _ in range(count)
        ]

    _last_audio_indices = set(selected)

    return selected
# ============================================================
# SINGLE OBSERVATION
# ============================================================

def run_observation(
    observation_number,
    thermal_path,
    radar_index,
    audio_index,
    models,
    X_test,
    y_test,
    distance_test,
    audio_samples,
):
    """
    Execute one complete SENSE-X multimodal observation.

    REAL:
        thermal inference
        radar inference
        audio inference
        OASF fusion
        localization
        priority
        active-sensing decision

    UAV physical movement remains simulated by the UAV controller.
    """

    print()
    print("=" * 70)
    print(
        f"OBSERVATION {observation_number}"
    )
    print("=" * 70)

    thermal_model = models["thermal"]
    radar_model = models["radar"]
    audio_model = models["audio"]

    fusion_engine = models["fusion"]
    localizer = models["localizer"]
    radar_range_lookup = models["radar_range"]
    priority_engine = models["priority"]
    active_controller = models["active"]


    # ========================================================
    # THERMAL
    # ========================================================

    if not thermal_path.exists():
        raise FileNotFoundError(
            f"Thermal image missing:\n"
            f"{thermal_path}"
        )

    thermal_image = cv2.imread(
        str(thermal_path),
        cv2.IMREAD_UNCHANGED,
    )

    if thermal_image is None:
        raise RuntimeError(
            f"OpenCV could not read thermal image:\n"
            f"{thermal_path}"
        )

    thermal_height = int(
        thermal_image.shape[0]
    )

    thermal_width = int(
        thermal_image.shape[1]
    )

    thermal_result = thermal_model.predict(
        thermal_image
    )

    thermal_evidence = safe_float(
        thermal_result.get(
            "human_probability",
            0.0,
        )
    )

    detections = safe_int(
        thermal_result.get(
            "detections",
            0,
        )
    )

    boxes = thermal_result.get(
        "boxes",
        [],
    ) or []

    best_box = None

    if boxes:
        best_box = max(
            boxes,
            key=lambda box: safe_float(
                box.get(
                    "confidence",
                    0.0,
                )
            ),
        )

    bbox = (
        best_box.get("bbox")
        if best_box
        else None
    )

    bbox_json = None

    if bbox is not None:
        bbox_json = [
            safe_float(value)
            for value in bbox
        ]

    best_confidence = (
        safe_float(
            best_box.get(
                "confidence",
                0.0,
            )
        )
        if best_box
        else 0.0
    )

    print(
        f"Thermal image: "
        f"{thermal_path.name}"
    )

    print(
        f"Thermal evidence: "
        f"{thermal_evidence:.4f}"
    )

    print(
        f"Thermal detections: "
        f"{detections}"
    )


    # ========================================================
    # RADAR
    # ========================================================

    if radar_index >= len(X_test):
        raise IndexError(
            f"Radar index {radar_index} "
            f"is outside test set."
        )

    radar_result = radar_model.predict(
        X_test[radar_index]
    )

    radar_evidence = safe_float(
        radar_result.get(
            "human_probability",
            0.0,
        )
    )

    radar_label = str(
        radar_result.get(
            "label",
            "unknown",
        )
    )

    radar_actual = safe_int(
        y_test[radar_index]
    )

    print(
        f"Radar evidence: "
        f"{radar_evidence:.4f}"
    )


    # ========================================================
    # AUDIO
    # ========================================================

    if audio_index >= len(audio_samples):
        raise IndexError(
            f"Audio index {audio_index} "
            f"is unavailable."
        )

    audio_sample = audio_samples[
        audio_index
    ]

    audio_result = run_audio_inference(
        audio_sample,
        audio_model,
    )

    audio_evidence = safe_float(
        audio_result.get(
            "source_present_probability",
            0.0,
        )
    )

    audio_label = str(
        audio_result.get(
            "label",
            "unknown",
        )
    )

    audio_clips = safe_int(
        audio_result.get(
            "clips_analyzed",
            0,
        )
    )

    audio_file = str(
        audio_sample.get(
            "file_path",
            "unknown",
        )
    )

    print(
        f"Audio evidence: "
        f"{audio_evidence:.4f}"
    )


    # ========================================================
    # OASF
    # ========================================================

    fusion_result = fusion_engine.fuse(
        thermal_evidence=thermal_evidence,
        radar_evidence=radar_evidence,
        audio_evidence=audio_evidence,
        occlusion=0.50,
        noise=0.30,
    )

    fusion_score = safe_float(
        fusion_result.human_score
    )

    uncertainty = safe_float(
        fusion_result.uncertainty
    )

    decision = str(
        fusion_result.decision
    )

    print(
        f"Fusion score: "
        f"{fusion_score:.4f}"
    )

    print(
        f"Uncertainty: "
        f"{uncertainty:.4f}"
    )

    print(
        f"Decision: {decision}"
    )


    # ========================================================
    # LOCALIZATION
    # ========================================================

    localization_json = None
    localization_confidence = 0.0

    # ========================================================
    # SAMPLE-ALIGNED RADAR RANGE
    # ========================================================

    radar_distance_raw = distance_test[radar_index]

    if np.isfinite(radar_distance_raw):
        radar_range = float(radar_distance_raw)

        print(
            f"Radar sample {radar_index} "
            f"dataset distance: "
            f"{radar_range:.2f} m"
        )

    else:
        radar_range = None

        print(
            f"Radar sample {radar_index} "
            "has no human-associated distance."
        )

    if bbox_json is not None:

        localization_result = (
            localizer.localize(
                bbox=bbox_json,
                image_width=thermal_width,
                image_height=thermal_height,
                fusion_confidence=fusion_score,
                radar_range_m=radar_range,
            )
        )

        localization_confidence = safe_float(
            localization_result.confidence
        )

        localization_json = {
            "x": safe_float(
                localization_result.x
            ),

            "y": safe_float(
                localization_result.y
            ),

            "z": safe_float(
                localization_result.z
            ),

            "bearing": safe_float(
                localization_result.bearing_deg
            ),

            "bearing_deg": safe_float(
                localization_result.bearing_deg
            ),

            "range": safe_float(
                localization_result.range_m
            ),

            "range_m": safe_float(
                localization_result.range_m
            ),

            "confidence": (
                localization_confidence
            ),

            "source": str(
                getattr(
                    localization_result,
                    "source",
                    "thermal_bbox + radar_range",
                )
            ),
        }

        print(
            f"Localization confidence: "
            f"{localization_confidence:.4f}"
        )

    else:
        print(
            "Localization unavailable: "
            "no thermal bounding box."
        )


    # ========================================================
    # PRIORITY
    # ========================================================

    priority_result = (
        priority_engine.calculate_priority(
            human_probability=fusion_score,
            urgency=0.80,
            distance_factor=0.70,
            obstruction=0.50,
        )
    )

    priority_score = safe_float(
        priority_result.priority_score
    )

    priority_level = str(
        priority_result.priority_level
    )

    recommended_action = str(
        priority_result.recommended_action
    )


    # ========================================================
    # ACTIVE SENSING
    # ========================================================

    active_result = (
        active_controller.decide(
            fusion_score=fusion_score,
            uncertainty=uncertainty,
            priority_level=priority_level,
            localization_confidence=(
                localization_confidence
            ),
        )
    )

    active_action = str(
        active_result.action
    )

    print(
        f"Active sensing: "
        f"{active_action}"
    )


    # ========================================================
    # THERMAL IMAGE URL
    # ========================================================

    thermal_url = (
        f"/thermal/"
        f"{thermal_path.name}"
    )


    # ========================================================
    # JSON
    # ========================================================

    return {
        "observation_number": (
            observation_number
        ),

        "thermal": {
            "human_probability": (
                thermal_evidence
            ),

            "evidence": (
                thermal_evidence
            ),

            "detections": detections,

            "boxes": boxes,

            "bbox": bbox_json,

            "best_bbox": bbox_json,

            "confidence": (
                best_confidence
            ),

            "image_width": (
                thermal_width
            ),

            "image_height": (
                thermal_height
            ),

            "image_name": (
                thermal_path.name
            ),

            "image_url": (
                thermal_url
            ),
        },

        "radar": {
            "human_probability": (
                radar_evidence
            ),

            "evidence": (
                radar_evidence
            ),

            "label": (
                radar_label
            ),

            "actual_label": (
                radar_actual
            ),

            "sample_index": (
                radar_index
            ),
        },

        "audio": {
            "source_present_probability": (
                audio_evidence
            ),

            "evidence": (
                audio_evidence
            ),

            "label": (
                audio_label
            ),

            "clips_analyzed": (
                audio_clips
            ),

            "file_path": (
                audio_file
            ),

            "sample_index": (
                audio_index
            ),
        },

        "fusion": {
            "human_score": (
                fusion_score
            ),

            "score": (
                fusion_score
            ),

            "uncertainty": (
                uncertainty
            ),

            "decision": (
                decision
            ),

            "thermal_weight": safe_float(
                fusion_result.thermal_weight
            ),

            "radar_weight": safe_float(
                fusion_result.radar_weight
            ),

            "audio_weight": safe_float(
                fusion_result.audio_weight
            ),

            "explanation": str(
                fusion_result.explanation
            ),

            "occlusion": 0.50,

            "noise": 0.30,
        },

        "localization": (
            localization_json
        ),

        "priority": {
            "priority_score": (
                priority_score
            ),

            "score": (
                priority_score
            ),

            "priority_level": (
                priority_level
            ),

            "level": (
                priority_level
            ),

            "recommended_action": (
                recommended_action
            ),
        },

        "active_sensing": {
            "action": (
                active_action
            ),

            "reason": str(
                active_result.reason
            ),

            "next_step": str(
                active_result.next_step
            ),
        },

        # Alias because some dashboard normalizers
        # may expect "active".
        "active": {
            "action": (
                active_action
            ),

            "reason": str(
                active_result.reason
            ),

            "next_step": str(
                active_result.next_step
            ),
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "online",
        "system": "SENSE-X",
        "dashboard": True,
        "mission_endpoint": (
            "/api/mission/run"
        ),
        "thermal_directory": (
            THERMAL_DIR.exists()
        ),
        "radar_data": (
            RADAR_X_TEST.exists()
            and RADAR_Y_TEST.exists()
        ),
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug/files")
async def debug_files():

    return {
        "project_root": (
            str(PROJECT_ROOT)
        ),

        "dashboard_dir": (
            str(DASHBOARD_DIR)
        ),

        "dashboard": {
            "index": INDEX_FILE.exists(),
            "css": STYLE_FILE.exists(),
            "javascript": JS_FILE.exists(),
        },

        "thermal": {
            "directory": THERMAL_DIR.exists(),

            "image_count": (
                len(get_thermal_candidates())
                if THERMAL_DIR.exists()
                else 0
            ),

            "dynamic_selection": True,
        },

        "radar": {
            "x_test": (
                RADAR_X_TEST.exists()
            ),

            "y_test": (
                RADAR_Y_TEST.exists()
            ),
        },
    }


# ============================================================
# REAL CLOSED-LOOP MISSION
# ============================================================

# ============================================================
# REAL CLOSED-LOOP MISSION
# ============================================================

@app.post("/api/mission/run")
async def run_mission():
    """
    Execute one SENSE-X sequential multimodal mission.

    Thermal, radar and audio observations are selected dynamically
    from their real datasets.

    Important scientific limitation:
    The three datasets are independent and are NOT temporally
    synchronized measurements of the same physical scene.

    UAV movement remains simulated.
    """

    try:
        # ====================================================
        # MISSION START
        # ====================================================

        print()
        print("=" * 70)
        print("SENSE-X DASHBOARD MISSION REQUEST")
        print("=" * 70)

        # ----------------------------------------------------
        # LOAD MODELS
        # ----------------------------------------------------

        models = load_models()

        # ----------------------------------------------------
        # LOAD RADAR DATA
        # ----------------------------------------------------

        X_test, y_test, distance_test = load_radar_data()

        # ----------------------------------------------------
        # LOAD AUDIO POOL
        # ----------------------------------------------------

        audio_samples = load_audio_samples(
            AUDIO_POOL_SIZE
        )

        # ====================================================
        # DYNAMIC REAL DATASET SELECTION
        # ====================================================

        thermal_samples = choose_thermal_samples(
            count=2
        )

        radar_indices = choose_radar_indices(
            dataset_size=len(X_test),
            count=2,
        )

        audio_indices = choose_audio_indices(
            number_of_samples=len(audio_samples),
            count=2,
        )

        print()
        print("-" * 70)
        print("DYNAMIC MISSION INPUTS")
        print("-" * 70)

        print(
            "Observation 1 thermal:",
            thermal_samples[0].name,
        )

        print(
            "Observation 1 radar index:",
            radar_indices[0],
        )

        print(
            "Observation 1 audio index:",
            audio_indices[0],
        )

        print(
            "Observation 2 thermal:",
            thermal_samples[1].name,
        )

        print(
            "Observation 2 radar index:",
            radar_indices[1],
        )

        print(
            "Observation 2 audio index:",
            audio_indices[1],
        )

        print("-" * 70)

        # ====================================================
        # UAV INITIAL STATE
        # ====================================================

        initial_position = UAVPosition(
            x=0.0,
            y=0.0,
            z=10.0,
        )

        uav_position = initial_position

        # ====================================================
        # OBSERVATION 1
        # ====================================================

        observation_1 = run_observation(
            observation_number=1,
            thermal_path=thermal_samples[0],
            radar_index=radar_indices[0],
            audio_index=audio_indices[0],
            models=models,
            X_test=X_test,
            y_test=y_test,
            distance_test=distance_test,
            audio_samples=audio_samples,
        )

        observation_2 = None
        movement_json = None

        first_action = (
            observation_1[
                "active_sensing"
            ]["action"]
        )

        # ====================================================
        # CLOSED-LOOP ACTIVE SENSING
        # ====================================================

        if first_action == "REPOSITION_AND_RESCAN":

            print()
            print(
                "Active sensing requested "
                "REPOSITION_AND_RESCAN."
            )

            uav_controller = models["uav"]

            # ------------------------------------------------
            # SIMULATED UAV REPOSITION
            # ------------------------------------------------

            movement = uav_controller.reposition(
                position=uav_position,
                direction="lateral",
            )

            uav_position = movement.new_position

            movement_json = {
                "action": str(
                    movement.action
                ),

                "direction": "lateral",

                "movement_distance": safe_float(
                    movement.movement_distance
                ),

                "old_position": {
                    "x": safe_float(
                        movement.old_position.x
                    ),
                    "y": safe_float(
                        movement.old_position.y
                    ),
                    "z": safe_float(
                        movement.old_position.z
                    ),
                },

                "new_position": {
                    "x": safe_float(
                        movement.new_position.x
                    ),
                    "y": safe_float(
                        movement.new_position.y
                    ),
                    "z": safe_float(
                        movement.new_position.z
                    ),
                },
            }

            # ================================================
            # OBSERVATION 2
            # ================================================

            observation_2 = run_observation(
                observation_number=2,
                thermal_path=thermal_samples[1],
                radar_index=radar_indices[1],
                audio_index=audio_indices[1],
                models=models,
                X_test=X_test,
                y_test=y_test,
                distance_test=distance_test,
                audio_samples=audio_samples,
            )

        else:
            print()
            print(
                "Observation 1 did not request "
                "REPOSITION_AND_RESCAN."
            )

            print(
                "Closed-loop mission terminating "
                "after Observation 1."
            )

        # ====================================================
        # FINAL OBSERVATION
        # ====================================================

        final_observation = (
            observation_2
            if observation_2 is not None
            else observation_1
        )

        final_action = (
            final_observation[
                "active_sensing"
            ]["action"]
        )

        final_decision = (
            final_observation[
                "fusion"
            ]["decision"]
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {
            "success": True,

            "system": "SENSE-X",

            "mission_id": "SX-001",

            "mode": "SEARCH & RESCUE",

            "data_mode": (
                "REAL DATASET INFERENCE"
            ),

            "temporally_synchronized": False,

            "uav_motion": "SIMULATED",

            # -----------------------------------------------
            # OBSERVATIONS
            # -----------------------------------------------

            "observation_1": observation_1,

            "observation_2": observation_2,

            # -----------------------------------------------
            # ACTIVE MOVEMENT
            # -----------------------------------------------

            "movement": movement_json,

            # -----------------------------------------------
            # UAV STATE
            # -----------------------------------------------

            "uav": {
                "initial_position": {
                    "x": safe_float(
                        initial_position.x
                    ),
                    "y": safe_float(
                        initial_position.y
                    ),
                    "z": safe_float(
                        initial_position.z
                    ),
                },

                "position": {
                    "x": safe_float(
                        uav_position.x
                    ),
                    "y": safe_float(
                        uav_position.y
                    ),
                    "z": safe_float(
                        uav_position.z
                    ),
                },

                "final_position": {
                    "x": safe_float(
                        uav_position.x
                    ),
                    "y": safe_float(
                        uav_position.y
                    ),
                    "z": safe_float(
                        uav_position.z
                    ),
                },
            },

            # -----------------------------------------------
            # FINAL RESULT
            # -----------------------------------------------

            "final": {
                "decision": final_decision,

                "action": final_action,

                "fusion_score": safe_float(
                    final_observation[
                        "fusion"
                    ]["human_score"]
                ),

                "uncertainty": safe_float(
                    final_observation[
                        "fusion"
                    ]["uncertainty"]
                ),

                "priority": (
                    final_observation[
                        "priority"
                    ]
                ),

                "localization": (
                    final_observation[
                        "localization"
                    ]
                ),
            },

            # -----------------------------------------------
            # MISSION INPUT TRACE
            # Useful for proving successive missions are
            # actually using different dataset samples.
            # -----------------------------------------------

            "mission_inputs": {
                "thermal": [
                    thermal_samples[0].name,
                    thermal_samples[1].name,
                ],

                "radar_indices": [
                    int(radar_indices[0]),
                    int(radar_indices[1]),
                ],

                "audio_indices": [
                    int(audio_indices[0]),
                    int(audio_indices[1]),
                ],
            },
        }

        # ====================================================
        # MISSION COMPLETE
        # ====================================================

        print()
        print("=" * 70)
        print("MISSION COMPLETE")
        print("=" * 70)

        print(
            f"Final decision: "
            f"{final_decision}"
        )

        print(
            f"Final action: "
            f"{final_action}"
        )

        print(
            "Final UAV position: "
            f"({uav_position.x:.2f}, "
            f"{uav_position.y:.2f}, "
            f"{uav_position.z:.2f})"
        )

        print()
        print("Mission inputs:")

        print(
            "  Thermal:",
            thermal_samples[0].name,
            "->",
            thermal_samples[1].name,
        )

        print(
            "  Radar:",
            radar_indices[0],
            "->",
            radar_indices[1],
        )

        print(
            "  Audio:",
            audio_indices[0],
            "->",
            audio_indices[1],
        )

        print("=" * 70)
        print()

        return response

    except Exception as exc:

        print()
        print("=" * 70)
        print("SENSE-X MISSION ERROR")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 70)
        print()

        raise HTTPException(
            status_code=500,
            detail={
                "error": (
                    type(exc).__name__
                ),

                "message": (
                    str(exc)
                ),
            },
        ) from exc


# ============================================================
# COMPATIBILITY API ROUTES
# ============================================================

# Your current app.js tries these if /api/mission/run fails.
# Keeping aliases makes the backend tolerant of older dashboard
# versions.

@app.post("/api/run-mission")
async def run_mission_compatibility():
    return await run_mission()


@app.post("/api/mission")
async def mission_compatibility():
    return await run_mission()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SENSE-X RESCUE COMMAND SERVER")
    print("=" * 70)

    print(
        "Dashboard: "
        "http://127.0.0.1:8000"
    )

    print(
        "API docs: "
        "http://127.0.0.1:8000/docs"
    )

    print(
        "Mission API: "
        "POST /api/mission/run"
    )

    print("=" * 70)
    print()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )