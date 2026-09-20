from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionResult:
    human_score: float
    uncertainty: float
    decision: str

    thermal_evidence: float
    radar_evidence: float
    audio_evidence: float

    thermal_weight: float
    radar_weight: float
    audio_weight: float

    explanation: str


class OASFEngine:
    """
    SENSE-X Adaptive Multimodal Sensor Fusion.

    OASF = Obstruction-Aware Sensor Fusion

    Combines thermal, radar and audio evidence while adapting
    sensor weights according to environmental conditions.
    """

    def __init__(self):
        # Base weights for normal conditions.
        self.base_weights = {
            "thermal": 0.30,
            "radar": 0.45,
            "audio": 0.25,
        }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def calculate_weights(
        self,
        occlusion: float = 0.0,
        noise: float = 0.0,
    ) -> dict[str, float]:

        occlusion = self._clamp(occlusion)
        noise = self._clamp(noise)

        thermal = self.base_weights["thermal"]
        radar = self.base_weights["radar"]
        audio = self.base_weights["audio"]

        # Heavy visual obstruction:
        # rely more strongly on radar.
        if occlusion >= 0.70:
            thermal = 0.20
            radar = 0.60
            audio = 0.20

        elif occlusion >= 0.50:
            thermal = 0.25
            radar = 0.55
            audio = 0.20

        # High acoustic noise makes audio less reliable.
        if noise >= 0.70:
            reduction = 0.10
            audio -= reduction
            radar += reduction

        elif noise >= 0.50:
            reduction = 0.05
            audio -= reduction
            radar += reduction

        # Normalize to exactly 1.
        total = thermal + radar + audio

        thermal /= total
        radar /= total
        audio /= total

        return {
            "thermal": thermal,
            "radar": radar,
            "audio": audio,
        }

    def fuse(
        self,
        thermal_evidence: float,
        radar_evidence: float,
        audio_evidence: float,
        occlusion: float = 0.0,
        noise: float = 0.0,
    ) -> FusionResult:

        thermal_evidence = self._clamp(thermal_evidence)
        radar_evidence = self._clamp(radar_evidence)
        audio_evidence = self._clamp(audio_evidence)

        weights = self.calculate_weights(
            occlusion=occlusion,
            noise=noise,
        )

        # Weighted multimodal evidence.
        score = (
            thermal_evidence * weights["thermal"]
            + radar_evidence * weights["radar"]
            + audio_evidence * weights["audio"]
        )

        # ---------------------------------------------------------
        # Cross-modal agreement bonus
        # ---------------------------------------------------------
        positive_modalities = sum(
            evidence >= 0.60
            for evidence in [
                thermal_evidence,
                radar_evidence,
                audio_evidence,
            ]
        )

        if positive_modalities >= 2:
            score += 0.05

        # ---------------------------------------------------------
        # Thermal-only false-positive suppression
        # ---------------------------------------------------------
        thermal_only_signal = (
            thermal_evidence >= 0.70
            and radar_evidence < 0.30
            and audio_evidence < 0.30
        )

        if thermal_only_signal:
            score *= 0.55

        score = self._clamp(score)

        # ---------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------
        uncertainty = self._calculate_uncertainty(
            thermal_evidence,
            radar_evidence,
            audio_evidence,
        )

        # ---------------------------------------------------------
        # Decision
        # ---------------------------------------------------------
        if score >= 0.75:
            decision = "HUMAN_CONFIRMED"

        elif score >= 0.50:
            decision = "HUMAN_SUSPECTED"

        elif score >= 0.30:
            decision = "UNCERTAIN"

        else:
            decision = "NO_HUMAN"

        explanation = self._build_explanation(
            thermal_evidence,
            radar_evidence,
            audio_evidence,
            weights,
            occlusion,
            noise,
            thermal_only_signal,
            positive_modalities,
        )

        return FusionResult(
            human_score=score,
            uncertainty=uncertainty,
            decision=decision,

            thermal_evidence=thermal_evidence,
            radar_evidence=radar_evidence,
            audio_evidence=audio_evidence,

            thermal_weight=weights["thermal"],
            radar_weight=weights["radar"],
            audio_weight=weights["audio"],

            explanation=explanation,
        )

    @staticmethod
    def _calculate_uncertainty(
        thermal: float,
        radar: float,
        audio: float,
    ) -> float:

        values = [thermal, radar, audio]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        # Scale standard deviation into [0, 1].
        uncertainty = variance ** 0.5

        return min(1.0, uncertainty)

    @staticmethod
    def _build_explanation(
        thermal: float,
        radar: float,
        audio: float,
        weights: dict[str, float],
        occlusion: float,
        noise: float,
        thermal_only_signal: bool,
        positive_modalities: int,
    ) -> str:

        reasons = []

        if occlusion >= 0.70:
            reasons.append(
                "Heavy visual obstruction increased radar reliance."
            )

        elif occlusion >= 0.50:
            reasons.append(
                "Moderate obstruction increased radar weighting."
            )

        if noise >= 0.70:
            reasons.append(
                "High acoustic noise reduced audio influence."
            )

        elif noise >= 0.50:
            reasons.append(
                "Elevated acoustic noise reduced audio influence."
            )

        if positive_modalities >= 2:
            reasons.append(
                "Multiple sensors independently support human presence."
            )

        if thermal_only_signal:
            reasons.append(
                "Thermal-only detection was suppressed because "
                "radar and audio did not support it."
            )

        if not reasons:
            strongest_sensor = max(
                {
                    "thermal": thermal,
                    "radar": radar,
                    "audio": audio,
                },
                key=lambda key: {
                    "thermal": thermal,
                    "radar": radar,
                    "audio": audio,
                }[key],
            )

            reasons.append(
                f"{strongest_sensor.capitalize()} provided the strongest evidence."
            )

        return " ".join(reasons)


# Singleton instance
_fusion_engine: Optional[OASFEngine] = None


def get_fusion_engine() -> OASFEngine:
    global _fusion_engine

    if _fusion_engine is None:
        _fusion_engine = OASFEngine()

    return _fusion_engine


def fuse_sensors(
    thermal_evidence: float,
    radar_evidence: float,
    audio_evidence: float,
    occlusion: float = 0.0,
    noise: float = 0.0,
) -> FusionResult:

    return get_fusion_engine().fuse(
        thermal_evidence=thermal_evidence,
        radar_evidence=radar_evidence,
        audio_evidence=audio_evidence,
        occlusion=occlusion,
        noise=noise,
    )