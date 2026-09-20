from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalizationResult:
    x: float
    y: float
    z: float
    range_m: Optional[float]
    bearing_deg: float
    confidence: float
    source: str


class HumanLocalizer:
    """
    Prototype relative human localization.

    Coordinate system:
        x = forward from UAV
        y = lateral position
        z = relative vertical position

    The current prototype uses:
        - thermal bounding-box position
        - radar range when available
        - fusion confidence
    """

    def __init__(
        self,
        camera_horizontal_fov_deg: float = 70.0,
        camera_vertical_fov_deg: float = 55.0,
    ):
        self.camera_horizontal_fov_deg = camera_horizontal_fov_deg
        self.camera_vertical_fov_deg = camera_vertical_fov_deg

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        return max(
            minimum,
            min(maximum, float(value)),
        )

    def localize(
        self,
        bbox,
        image_width: int,
        image_height: int,
        fusion_confidence: float,
        radar_range_m: Optional[float] = None,
    ) -> LocalizationResult:

        if bbox is None:
            return LocalizationResult(
                x=0.0,
                y=0.0,
                z=0.0,
                range_m=radar_range_m,
                bearing_deg=0.0,
                confidence=0.0,
                source="no_detection",
            )

        x1, y1, x2, y2 = bbox

        # Bounding-box center
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Normalize image coordinates to [-1, 1]
        horizontal_offset = (
            (center_x / image_width) - 0.5
        ) * 2.0

        vertical_offset = (
            (center_y / image_height) - 0.5
        ) * 2.0

        # Convert horizontal image position to bearing.
        bearing_deg = (
            horizontal_offset
            * (self.camera_horizontal_fov_deg / 2.0)
        )

        # If radar gives range, use it.
        # Otherwise use a conservative prototype range.
        if radar_range_m is not None:
            range_m = max(float(radar_range_m), 0.1)
        else:
            range_m = 5.0

        # Convert polar horizontal position
        # into UAV-relative coordinates.
        import math

        bearing_rad = math.radians(bearing_deg)

        forward_x = (
            range_m
            * math.cos(bearing_rad)
        )

        lateral_y = (
            range_m
            * math.sin(bearing_rad)
        )

        # Vertical position is currently estimated
        # from image position.
        vertical_z = (
            -vertical_offset
            * range_m
            * 0.5
        )

        confidence = self._clamp(
            fusion_confidence
        )

        if radar_range_m is not None:
            source = "thermal_bbox + radar_range"
            confidence = self._clamp(
                confidence + 0.10
            )
        else:
            source = "thermal_bbox"

        return LocalizationResult(
            x=forward_x,
            y=lateral_y,
            z=vertical_z,
            range_m=range_m,
            bearing_deg=bearing_deg,
            confidence=confidence,
            source=source,
        )


_localizer = None


def get_human_localizer():
    global _localizer

    if _localizer is None:
        _localizer = HumanLocalizer()

    return _localizer