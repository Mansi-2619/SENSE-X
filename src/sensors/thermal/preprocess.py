import cv2
import numpy as np


def load_thermal_image(image_path):
    """
    Load thermal image.

    FLIR thermal images may contain 8-bit or higher-bit-depth
    representations, so preserve the original depth initially.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load thermal image: {image_path}"
        )

    return image


def normalize_thermal(image):
    """
    Normalize thermal intensity to 0-1.

    Uses the observed image range rather than assuming
    a fixed temperature scale.
    """

    image = image.astype(
        np.float32
    )

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum - minimum < 1e-8:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    normalized = (
        image - minimum
    ) / (
        maximum - minimum
    )

    return normalized


def preprocess_thermal(
    image,
    target_size=(640, 640)
):
    """
    Resize and normalize thermal image.
    """

    normalized = normalize_thermal(
        image
    )

    resized = cv2.resize(
        normalized,
        target_size
    )

    return resized.astype(
        np.float32
    )


def get_thermal_shape(image):
    """
    Return thermal image dimensions.
    """

    height, width = image.shape[:2]

    return {
        "width": width,
        "height": height,
        "channels": 1
    }