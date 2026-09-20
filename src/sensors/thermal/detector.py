import numpy as np


class ThermalPersonDetector:

    def __init__(self):

        self.model = None

    def load_model(self, model_path):

        self.model = model_path

    def predict(self, image):

        if self.model is None:

            raise RuntimeError(
                "Thermal detector model has not been loaded."
            )

        raise NotImplementedError(
            "Thermal model inference is not implemented yet."
        )