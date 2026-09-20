from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


MODEL_PATH = Path("models/audio/audio_source_detector_v2.pt")
CONFIG_PATH = Path("models/audio/audio_source_detector_v2_config.json")


class AudioCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AudioDetector:

    def __init__(self, model_path=MODEL_PATH):

        self.device = torch.device("cpu")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.model = AudioCNN().to(self.device)

        if isinstance(checkpoint, dict):

            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
            )

            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(checkpoint)

        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def predict(self, audio_path):

        waveform, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        duration = 3 * sr

        if len(waveform) < duration:
            waveform = np.pad(
                waveform,
                (0, duration - len(waveform))
            )
        else:
            waveform = waveform[:duration]

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_mels=64,
            n_fft=1024,
            hop_length=256
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():

            logits = self.model(
                tensor.to(self.device)
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        source_probability = float(
            probabilities[0, 1]
        )

        return {
            "available": True,
            "source_probability": source_probability
        }