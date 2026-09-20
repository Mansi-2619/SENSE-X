import numpy as np


class RGBPersonDetector:

    def __init__(self):
        """
        RGB person detector interface.

        The actual trained detector will be loaded here
        in the next stage.
        """

        self.model = None

    def load_model(self, model_path):
        """
        Load the trained RGB detector.

        Placeholder until the detector model is trained.
        """

        self.model = model_path

    def predict(self, image):
        """
        Return person detection probability.

        This function will be connected to the actual
        detector after training.
        """

        if self.model is None:

            raise RuntimeError(
                "RGB detector model has not been loaded."
            )

        # Actual inference will be implemented here.
        raise NotImplementedError(
            "RGB model inference is not implemented yet."
        )