from dataclasses import dataclass


@dataclass
class ActiveSensingResult:
    action: str
    reason: str
    next_step: str


class ActiveSensingController:
    """
    Decides whether the UAV should investigate immediately,
    reposition and rescan, or continue searching.

    This is a mission-level decision layer.
    It does not physically control a UAV yet.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        uncertainty_threshold: float = 0.25,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def decide(
        self,
        fusion_score: float,
        uncertainty: float,
        priority_level: str,
        localization_confidence: float,
    ) -> ActiveSensingResult:

        fusion_score = max(0.0, min(1.0, float(fusion_score)))
        uncertainty = max(0.0, min(1.0, float(uncertainty)))
        localization_confidence = max(
            0.0,
            min(1.0, float(localization_confidence)),
        )

        # High-confidence target with reliable localization
        if (
            fusion_score >= self.confidence_threshold
            and uncertainty <= self.uncertainty_threshold
            and localization_confidence >= 0.70
        ):
            return ActiveSensingResult(
                action="INVESTIGATE",
                reason=(
                    "High-confidence human evidence with "
                    "sufficient localization confidence."
                ),
                next_step=(
                    "Maintain target tracking and prioritize "
                    "investigation."
                ),
            )

        # Evidence exists, but uncertainty is too high
        if (
            fusion_score >= 0.40
            and (
                uncertainty > self.uncertainty_threshold
                or localization_confidence < 0.70
            )
        ):
            return ActiveSensingResult(
                action="REPOSITION_AND_RESCAN",
                reason=(
                    "Human evidence is present but uncertainty "
                    "or localization confidence is insufficient."
                ),
                next_step=(
                    "Change UAV viewpoint and acquire another "
                    "multimodal observation."
                ),
            )

        # Weak evidence
        return ActiveSensingResult(
            action="CONTINUE_SEARCH",
            reason=(
                "Current multimodal evidence is insufficient "
                "to prioritize a target."
            ),
            next_step=(
                "Continue scanning the environment for "
                "additional evidence."
            ),
        )


_controller = None


def get_active_sensing_controller():
    global _controller

    if _controller is None:
        _controller = ActiveSensingController()

    return _controller