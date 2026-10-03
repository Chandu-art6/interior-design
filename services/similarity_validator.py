"""
Design Similarity & Visual Diversity Validation Service
Ensures generated room redesigns are substantially different from the uploaded image
and distinctly unique from each other while preserving physical room architecture.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import cv2
import logging

from config import settings

logger = logging.getLogger("SIMILARITY_VALIDATOR")


class DesignSimilarityValidator:
    """
    Validates that generated interior redesigns achieve substantial visual and structural diversity
    compared to the original room photo, preventing near-duplicate or simple recoloring outputs.
    """

    def __init__(self, default_threshold: float = 0.65):
        self.default_threshold = getattr(settings, "SIMILARITY_THRESHOLD", default_threshold)

    def _normalize_image_for_analysis(self, image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Converts PIL image to standard RGB numpy array resized for fast metric comparison."""
        resized = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        return np.array(resized)

    def compute_similarity(
        self,
        original_image: Image.Image,
        redesign_image: Image.Image,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes composite multi-dimensional similarity score between original room and redesign.
        
        Metrics evaluated:
        1. Color distribution correlation (HSV/RGB 3D histograms)
        2. Structural pixel intensity distribution (L1 norm on downscaled luminance)
        3. Gradient / Edge feature similarity (Sobel spatial high-pass response)
        
        Returns:
            Dict containing similarity_score (0.0 to 1.0), diversity_score (0.0 to 1.0),
            and boolean is_sufficiently_different.
        """
        thresh = threshold if threshold is not None else self.default_threshold

        orig_np = self._normalize_image_for_analysis(original_image)
        rede_np = self._normalize_image_for_analysis(redesign_image)

        # 1. Color Histogram Correlation in HSV space
        orig_hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
        rede_hsv = cv2.cvtColor(rede_np, cv2.COLOR_RGB2HSV)

        hist_orig = cv2.calcHist([orig_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist_rede = cv2.calcHist([rede_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])

        cv2.normalize(hist_orig, hist_orig, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_rede, hist_rede, 0, 1, cv2.NORM_MINMAX)

        color_corr = float(cv2.compareHist(hist_orig, hist_rede, cv2.HISTCMP_CORREL))
        # Clamp correlation [-1, 1] to [0, 1]
        color_sim = max(0.0, min(1.0, (color_corr + 1.0) / 2.0))

        # 2. Structural Pixel Difference (Luminance Mean Absolute Difference)
        orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        rede_gray = cv2.cvtColor(rede_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        mean_abs_diff = float(np.mean(np.abs(orig_gray - rede_gray)))  # Range 0 to 255
        pixel_dissimilarity = min(1.0, mean_abs_diff / 80.0)  # ~80px avg diff = 100% distinct
        pixel_sim = 1.0 - pixel_dissimilarity

        # 3. Spatial Edge / High-Frequency Gradient Similarity
        orig_edges = cv2.Canny(orig_np, 50, 150).astype(np.float32)
        rede_edges = cv2.Canny(rede_np, 50, 150).astype(np.float32)
        edge_diff = float(np.mean(np.abs(orig_edges - rede_edges)) / 255.0)
        edge_sim = max(0.0, 1.0 - edge_diff * 2.0)

        # Composite Weighted Similarity
        # Weight: 45% Pixel Layout & Luminance + 35% Color Palette + 20% Edge Geometry
        composite_sim = round(float(0.45 * pixel_sim + 0.35 * color_sim + 0.20 * edge_sim), 3)
        diversity_score = round(max(0.0, min(1.0, 1.0 - composite_sim)), 3)

        is_different = bool(composite_sim <= thresh)

        return {
            "similarity_score": composite_sim,
            "diversity_score": diversity_score,
            "is_sufficiently_different": is_different,
            "threshold_used": thresh,
            "color_similarity": round(color_sim, 3),
            "spatial_similarity": round(pixel_sim, 3),
            "edge_similarity": round(edge_sim, 3),
            "diversity_status": "✅ Substantially Redesigned" if is_different else "⚠️ Too Similar to Original"
        }

    def compute_pairwise_diversity(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Computes pairwise diversity between all generated redesign options to guarantee
        that Option 1, Option 2, Option 3, Option 4 do not duplicate each other.
        """
        n = len(images)
        results = []
        for i in range(n):
            for j in range(i + 1, n):
                res = self.compute_similarity(images[i], images[j])
                results.append({
                    "pair": (i, j),
                    "pair_label": f"Option {i+1} vs Option {j+1}",
                    "similarity_score": res["similarity_score"],
                    "diversity_score": res["diversity_score"],
                    "is_distinct": res["is_sufficiently_different"]
                })
        return results


similarity_validator = DesignSimilarityValidator()
