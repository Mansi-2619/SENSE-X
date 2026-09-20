from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


MODEL_PATH = Path("models/radar/radar_human_detector.pt")


class RadarCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class RadarDetector:

    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cpu")
        self.model = RadarCNN().to(self.device)

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def predict(self, radar_frame):

        x = np.asarray(radar_frame, dtype=np.float32)

        if x.shape != (4, 180):
            raise ValueError(
                f"Expected radar input shape (4, 180), got {x.shape}"
            )

        tensor = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        human_probability = float(probabilities[0, 1])

        return {
            "available": True,
            "human_probability": human_probability
        }