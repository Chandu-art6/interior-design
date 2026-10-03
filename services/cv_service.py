"""
Computer Vision Service
Handles edge detection, room geometry analysis, depth estimation preprocessing, and structural line extraction.
"""
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


class CVService:
    """Provides OpenCV image processing utilities for ControlNet conditioning."""

    def __init__(self):
        pass

    def get_canny_edges(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Extracts Canny edge map from an input PIL image for ControlNet Canny model."""
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges)

    def extract_architectural_lines(self, image: Image.Image) -> Image.Image:
        """Extracts straight structural lines (M-LSD placeholder / Hough lines)."""
        # Logic to be expanded in Step 2
        return self.get_canny_edges(image, 50, 150)


cv_service = CVService()
