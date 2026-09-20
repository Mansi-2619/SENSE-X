from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "audio"
    / "audio_source_detector_v2.pt"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "audio"
    / "audio_source_detector_v2_config.json"
)

SAMPLE_RATE = 16000
N_MELS = 64

# These are read from the training script's preprocessing.
# We will use the model config if available for clip-related
# parameters, while keeping the feature extraction identical.
DEFAULT_CLIP_SECONDS = 3.0
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256


# ============================================================
# EXACT TRAINING ARCHITECTURE
# ============================================================

class AudioCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# AUDIO MODEL
# ============================================================

class AudioModel:

    def __init__(
        self,
        model_path=MODEL_PATH,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=DEFAULT_N_FFT,
        hop_length=DEFAULT_HOP_LENGTH,
        clip_seconds=DEFAULT_CLIP_SECONDS,
    ):

        self.model_path = Path(model_path)

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.clip_seconds = clip_seconds

        self.clip_samples = int(
            self.sample_rate * self.clip_seconds
        )

        self.device = torch.device("cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Audio model not found: {self.model_path}"
            )

        self.model = AudioCNN()

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # Handle both a raw state_dict and a checkpoint dictionary.
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    # ========================================================
    # AUDIO PREPROCESSING
    # ========================================================

    def prepare_audio(
        self,
        audio,
        sample_rate
    ):

        y = np.asarray(
            audio,
            dtype=np.float32
        )

        # Convert multichannel -> mono
        if y.ndim > 1:

            y = np.mean(
                y,
                axis=-1
            )

        # Resample if necessary
        if sample_rate != self.sample_rate:

            y = librosa.resample(
                y=y,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )

        # Remove DC offset
        y = y - np.mean(y)

        return y.astype(
            np.float32
        )

    # ========================================================
    # CLIP EXTRACTION
    # ========================================================

    def extract_clips(
        self,
        y,
        number_of_clips=1
    ):

        if len(y) < self.clip_samples:

            return [
                np.pad(
                    y,
                    (
                        0,
                        self.clip_samples - len(y)
                    )
                )
            ]

        max_start = (
            len(y) - self.clip_samples
        )

        if max_start == 0:

            starts = [0]

        else:

            starts = np.linspace(
                0,
                max_start,
                number_of_clips
            ).astype(int)

        clips = []

        for start in starts:

            clip = y[
                start:
                start + self.clip_samples
            ]

            if len(clip) < self.clip_samples:

                clip = np.pad(
                    clip,
                    (
                        0,
                        self.clip_samples - len(clip)
                    )
                )

            clips.append(
                clip
            )

        return clips

    # ========================================================
    # MEL FEATURE EXTRACTION
    # ========================================================

    def extract_mel(
        self,
        y
    ):

        if len(y) < self.clip_samples:

            y = np.pad(
                y,
                (
                    0,
                    self.clip_samples - len(y)
                )
            )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Same per-example normalization used during training.
        mean = mel_db.mean()
        std = mel_db.std()

        if std > 1e-6:

            mel_db = (
                mel_db - mean
            ) / std

        else:

            mel_db = (
                mel_db - mean
            )

        return mel_db.astype(
            np.float32
        )

    # ========================================================
    # SINGLE CLIP PREDICTION
    # ========================================================

    def predict_clip(
        self,
        clip
    ):

        mel = self.extract_mel(
            clip
        )

        tensor = torch.from_numpy(
            mel[None, None, :, :]
        ).float().to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        drone_only_probability = float(
            probabilities[0].item()
        )

        source_present_probability = float(
            probabilities[1].item()
        )

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        return {
            "source_present_probability":
                source_present_probability,

            "drone_only_probability":
                drone_only_probability,

            "prediction":
                prediction,

            "label":
                (
                    "source_present"
                    if prediction == 1
                    else "drone_only"
                )
        }

    # ========================================================
    # FULL AUDIO PREDICTION
    # ========================================================

    def predict(
        self,
        audio,
        sample_rate=None,
        number_of_clips=3
    ):

        if sample_rate is None:

            sample_rate = self.sample_rate

        y = self.prepare_audio(
            audio,
            sample_rate
        )

        clips = self.extract_clips(
            y,
            number_of_clips
        )

        predictions = []

        for clip in clips:

            predictions.append(
                self.predict_clip(
                    clip
                )
            )

        source_scores = [
            item["source_present_probability"]
            for item in predictions
        ]

        drone_scores = [
            item["drone_only_probability"]
            for item in predictions
        ]

        source_probability = float(
            np.mean(
                source_scores
            )
        )

        drone_probability = float(
            np.mean(
                drone_scores
            )
        )

        prediction = int(
            source_probability >= drone_probability
        )

        return {
            "source_present_probability":
                source_probability,

            "drone_only_probability":
                drone_probability,

            "prediction":
                prediction,

            "label":
                (
                    "source_present"
                    if prediction == 1
                    else "drone_only"
                ),

            "clips_analyzed":
                len(predictions),

            "clip_predictions":
                predictions
        }


# ============================================================
# SINGLETON
# ============================================================

_audio_model = None


def get_audio_model():

    global _audio_model

    if _audio_model is None:

        _audio_model = AudioModel()

    return _audio_model


def predict_audio(
    audio,
    sample_rate=16000,
    number_of_clips=3
):

    model = get_audio_model()

    return model.predict(
        audio,
        sample_rate=sample_rate,
        number_of_clips=number_of_clips
    )