from dataclasses import dataclass
import math


@dataclass
class UAVPosition:
    x: float
    y: float
    z: float


@dataclass
class UAVAction:
    action: str
    old_position: UAVPosition
    new_position: UAVPosition
    movement_distance: float


class UAVController:
    """
    Lightweight UAV movement simulator.

    This does not control real hardware.
    It provides the closed-loop mission simulation for SENSE-X.
    """

    def __init__(self, step_size_m: float = 2.0):
        self.step_size_m = step_size_m

    def reposition(
        self,
        position: UAVPosition,
        direction: str = "lateral",
    ) -> UAVAction:

        old_position = UAVPosition(
            position.x,
            position.y,
            position.z,
        )

        if direction == "lateral":
            new_position = UAVPosition(
                position.x,
                position.y + self.step_size_m,
                position.z,
            )

        elif direction == "forward":
            new_position = UAVPosition(
                position.x + self.step_size_m,
                position.y,
                position.z,
            )

        elif direction == "backward":
            new_position = UAVPosition(
                position.x - self.step_size_m,
                position.y,
                position.z,
            )

        elif direction == "vertical":
            new_position = UAVPosition(
                position.x,
                position.y,
                position.z + self.step_size_m,
            )

        else:
            raise ValueError(
                f"Unsupported movement direction: {direction}"
            )

        distance = math.sqrt(
            (new_position.x - old_position.x) ** 2
            + (new_position.y - old_position.y) ** 2
            + (new_position.z - old_position.z) ** 2
        )

        return UAVAction(
            action="REPOSITION",
            old_position=old_position,
            new_position=new_position,
            movement_distance=distance,
        )


_controller = None


def get_uav_controller():
    global _controller

    if _controller is None:
        _controller = UAVController()

    return _controller