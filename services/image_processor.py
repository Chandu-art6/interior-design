"""
Image Preprocessing Service
Provides high-performance image standardization, edge-preserving bilateral filtering,
and K-Means dominant color palette extraction for interior redesign pipelines.
"""
from typing import Tuple, List, Dict, Any, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    """Handles image standardization, noise reduction, and dominant color palette extraction."""

    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        self.target_size = target_size

    def load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Loads and converts various image inputs into a standard RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 4:
                # RGBA to RGB
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB))
            else:
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        return image

    def standardize_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        target_size: Tuple[int, int] = (1024, 1024),
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resizes and standardizes uploaded room images to 1024x1024 resolution.
        If maintain_aspect_ratio is True, adds symmetric letterbox padding to avoid distortion.
        """
        image = self.load_image(image_input)
        target_w, target_h = target_size

        if not maintain_aspect_ratio:
            return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        orig_w, orig_h = image.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create target canvas with neutral dark/black padding
        padded_canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded_canvas.paste(resized_img, (paste_x, paste_y))

        return padded_canvas

    def apply_bilateral_filter(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0
    ) -> Image.Image:
        """
        Applies OpenCV Bilateral Filtering for edge-preserving noise reduction.
        Smooths wall textures and sensor noise while retaining crisp structural room edges.
        """
        image = self.load_image(image_input)
        img_np = np.array(image)

        # Convert RGB to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # Convert back to RGB PIL Image
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

    def extract_dominant_colors(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extracts dominant color palettes using OpenCV K-Means clustering.
        Returns a sorted list of dictionaries containing RGB tuples, HEX codes, and pixel percentage.
        """
        image = self.load_image(image_input)
        
        # Downsample image for rapid K-Means computation while preserving color distribution
        small_img = image.resize((150, 150), Image.Resampling.BILINEAR)
        img_np = np.array(small_img).reshape((-1, 3)).astype(np.float32)

        # K-Means clustering criteria: 100 max iterations or 0.2 epsilon accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS

        compactness, labels, centers = cv2.kmeans(
            img_np, k, None, criteria, 10, flags
        )

        labels = labels.flatten()
        total_pixels = len(labels)

        # Count frequencies of each cluster
        palette = []
        for i, center in enumerate(centers):
            count = int(np.sum(labels == i))
            percentage = round((count / total_pixels) * 100, 2)
            
            r, g, b = int(round(center[0])), int(round(center[1])), int(round(center[2]))
            hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
            
            palette.append({
                "hex": hex_code,
                "rgb": [r, g, b],
                "percentage": percentage
            })

        # Sort dominant colors by highest coverage percentage
        palette.sort(key=lambda x: x["percentage"], reverse=True)
        return palette

    def preprocess_upload(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        target_size: Tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """
        Standardizes an uploaded room image to 1024x1024 resolution.
        """
        return self.standardize_image(image_input, target_size=target_size, maintain_aspect_ratio=True)

    def extract_dominant_palette(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        num_colors: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extracts dominant color palettes using K-Means clustering.
        """
        return self.extract_dominant_colors(image_input, k=num_colors)

    def preprocess_pipeline(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        target_size: Tuple[int, int] = (1024, 1024),
        palette_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Executes the full preprocessing pipeline:
        1. Standardization to 1024x1024.
        2. Bilateral edge-preserving filtering.
        3. Dominant color extraction.
        """
        standardized_img = self.standardize_image(image_input, target_size=target_size)
        filtered_img = self.apply_bilateral_filter(standardized_img)
        dominant_palette = self.extract_dominant_colors(standardized_img, k=palette_clusters)

        return {
            "standardized_image": standardized_img,
            "filtered_image": filtered_img,
            "dominant_palette": dominant_palette,
            "dimensions": target_size
        }



# Singleton image processor instance
image_processor = ImageProcessor()
