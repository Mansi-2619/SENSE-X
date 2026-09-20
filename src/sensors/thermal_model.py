from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE
    / "runs"
    / "detect"
    / "models"
    / "thermal"
    / "flir_thermal_fast"
    / "weights"
    / "best.pt"
)


# ============================================================
# THERMAL MODEL
# ============================================================

class ThermalModel:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Thermal model not found:\n{MODEL_PATH}"
            )

        print(
            f"Loading thermal model:\n{MODEL_PATH}"
        )

        self.model = YOLO(
            str(MODEL_PATH)
        )

        print(
            "Thermal model loaded successfully."
        )

    # --------------------------------------------------------
    # PREDICT FROM IMAGE
    # --------------------------------------------------------

    def predict(
        self,
        image,
        conf_threshold=0.25,
    ):

        """
        Run thermal person detection.

        Parameters
        ----------
        image:
            Thermal image as a numpy array.

        conf_threshold:
            Minimum YOLO detection confidence.

        Returns
        -------
        dict
            Standardized thermal result.
        """

        if image is None:

            raise ValueError(
                "Thermal image is None."
            )

        if not isinstance(
            image,
            np.ndarray
        ):

            raise TypeError(
                "Thermal image must be "
                "a numpy array."
            )

        # ----------------------------------------------------
        # Handle grayscale thermal images
        # ----------------------------------------------------

        if len(image.shape) == 2:

            # YOLO expects 3-channel image
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        elif (
            len(image.shape) == 3
            and image.shape[2] == 1
        ):

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        elif (
            len(image.shape) != 3
            or image.shape[2] != 3
        ):

            raise ValueError(
                "Thermal image must have "
                "shape (H,W) or (H,W,3)."
            )

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False,
            device="cpu",
        )

        result = results[0]

        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                confidence = float(
                    box.conf[0].item()
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                detections.append(
                    {
                        "confidence": confidence,
                        "bbox": xyxy,
                    }
                )

        # ----------------------------------------------------
        # Calculate human probability
        # ----------------------------------------------------

        if detections:

            confidences = [
                d["confidence"]
                for d in detections
            ]

            # Use strongest detected person
            human_probability = max(
                confidences
            )

        else:

            human_probability = 0.0

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        return {
            "human_probability":
                float(human_probability),

            "detections":
                len(detections),

            "confidence":
                float(human_probability),

            "detected":
                len(detections) > 0,

            "boxes":
                detections,
        }


# ============================================================
# SINGLETON
# ============================================================

_thermal_model = None


def get_thermal_model():

    global _thermal_model

    if _thermal_model is None:

        _thermal_model = ThermalModel()

    return _thermal_model


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def predict_thermal(
    image,
    conf_threshold=0.25,
):

    model = get_thermal_model()

    return model.predict(
        image,
        conf_threshold=conf_threshold,
    )