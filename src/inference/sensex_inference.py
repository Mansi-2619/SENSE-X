from pathlib import Path

from src.inference.rgb_detector import RGBDetector
from src.inference.thermal_detector import ThermalDetector
from src.inference.radar_detector import RadarDetector
from src.inference.audio_detector import AudioDetector


class SENSEXInference:

    def __init__(self):

        self.rgb = RGBDetector()
        self.thermal = ThermalDetector()

        self.radar = RadarDetector()
        self.audio = AudioDetector()

    def predict(
        self,
        rgb_path=None,
        thermal_path=None,
        radar_frame=None,
        audio_path=None
    ):

        rgb_result = (
            self.rgb.predict(rgb_path)
            if rgb_path
            else {
                "available": False,
                "person_probability": 0.0
            }
        )

        thermal_result = (
            self.thermal.predict(thermal_path)
            if thermal_path
            else {
                "available": False,
                "person_probability": 0.0
            }
        )

        radar_result = (
            self.radar.predict(radar_frame)
            if radar_frame is not None
            else {
                "available": False,
                "human_probability": 0.0
            }
        )

        audio_result = (
            self.audio.predict(audio_path)
            if audio_path
            else {
                "available": False,
                "source_probability": 0.0
            }
        )

        return {
            "rgb": rgb_result,
            "thermal": thermal_result,
            "radar": radar_result,
            "audio": audio_result
        }