from dataclasses import dataclass


@dataclass
class PriorityResult:
    priority_score: float
    priority_level: str
    recommended_action: str


class PriorityEngine:
    """
    Converts fused human evidence and mission context
    into an operational priority.
    """

    def __init__(self):
        self.human_weight = 0.65
        self.urgency_weight = 0.15
        self.distance_weight = 0.10
        self.obstruction_weight = 0.10

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0):
        return max(minimum, min(maximum, float(value)))

    def calculate_priority(
        self,
        human_probability: float,
        urgency: float = 0.0,
        distance_factor: float = 0.0,
        obstruction: float = 0.0,
    ) -> PriorityResult:

        human_probability = self._clamp(human_probability)
        urgency = self._clamp(urgency)
        distance_factor = self._clamp(distance_factor)
        obstruction = self._clamp(obstruction)

        score = (
            self.human_weight * human_probability
            + self.urgency_weight * urgency
            + self.distance_weight * distance_factor
            + self.obstruction_weight * obstruction
        )

        score = self._clamp(score)

        if score >= 0.82:
            level = "CRITICAL"
            action = "Immediate UAV attention required."

        elif score >= 0.65:
            level = "HIGH"
            action = "Prioritize target for investigation."

        elif score >= 0.45:
            level = "MEDIUM"
            action = "Monitor target and gather additional evidence."

        else:
            level = "LOW"
            action = "Low-priority target."

        return PriorityResult(
            priority_score=score,
            priority_level=level,
            recommended_action=action,
        )


_priority_engine = None


def get_priority_engine():
    global _priority_engine

    if _priority_engine is None:
        _priority_engine = PriorityEngine()

    return _priority_engine


def calculate_priority(
    human_probability: float,
    urgency: float = 0.0,
    distance_factor: float = 0.0,
    obstruction: float = 0.0,
):
    engine = get_priority_engine()

    return engine.calculate_priority(
        human_probability=human_probability,
        urgency=urgency,
        distance_factor=distance_factor,
        obstruction=obstruction,
    )