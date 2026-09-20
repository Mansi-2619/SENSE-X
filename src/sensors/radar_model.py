from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE
    / "models"
    / "radar"
    / "radar_human_detector.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# EXACT ARCHITECTURE USED FOR TRAINING
# ============================================================

class RadarHumanDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 4 -> 32
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 32 -> 64
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 64 -> 128
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 128 -> 128
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# RADAR MODEL WRAPPER
# ============================================================

class RadarModel:

    def __init__(self):

        self.device = DEVICE

        self.model = RadarHumanDetector().to(
            self.device
        )

        self._load_model()

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    def _load_model(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Radar model not found:\n{MODEL_PATH}"
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self, radar_input):

        """
        Predict human presence from one radar observation.

        Expected input:
            numpy array with shape (4, 180)

        Channels:
            0 = normalized magnitude
            1 = real
            2 = imaginary
            3 = phase

        Returns:
            {
                "human_probability": float,
                "no_human_probability": float,
                "prediction": int,
                "label": str
            }
        """

        # Convert to numpy
        x = np.asarray(
            radar_input,
            dtype=np.float32
        )

        # Validate shape
        if x.shape != (4, 180):

            raise ValueError(
                "Radar input must have shape "
                f"(4, 180), received {x.shape}"
            )

        # Add batch dimension
        x = torch.from_numpy(x).unsqueeze(0)

        x = x.to(self.device)

        # Model inference
        logits = self.model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        no_human_probability = (
            probabilities[0, 0].item()
        )

        human_probability = (
            probabilities[0, 1].item()
        )

        prediction = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        label = (
            "human"
            if prediction == 1
            else "no_human"
        )

        return {
            "human_probability": human_probability,
            "no_human_probability": no_human_probability,
            "prediction": prediction,
            "label": label,
        }


# ============================================================
# SINGLETON MODEL
# ============================================================

_radar_model = None


def get_radar_model():

    global _radar_model

    if _radar_model is None:

        _radar_model = RadarModel()

    return _radar_model


# ============================================================
# SIMPLE PUBLIC FUNCTION
# ============================================================

def predict_radar(radar_input):

    """
    Convenience function.

    Example:

        result = predict_radar(X)

        print(result["human_probability"])
    """

    model = get_radar_model()

    return model.predict(radar_input)