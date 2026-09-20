import cv2
import numpy as np


def load_rgb_image(image_path):
    """
    Load RGB image from disk.

    Returns:
        numpy array in RGB format
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load RGB image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def preprocess_rgb(image, target_size=(640, 640)):
    """
    Resize RGB image while keeping a simple preprocessing pipeline.
    """

    image = cv2.resize(
        image,
        target_size
    )

    image = image.astype(
        np.float32
    ) / 255.0

    return image


def get_rgb_shape(image):
    """
    Return image dimensions.
    """

    height, width = image.shape[:2]

    return {
        "width": width,
        "height": height,
        "channels": (
            image.shape[2]
            if image.ndim == 3
            else 1
        )
    }