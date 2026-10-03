"""
Room Segmenter & Binary Mask Generator Module
Generates high-precision binary masks for targeted inpainting.
Isolates items marked for removal/replacement while preserving walls, windows, and structural planes.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2

from config import settings


class RoomSegmenter:
    """
    Precision Mask Generator for Room Inpainting.
    Creates binary masks (255 = inpaint region, 0 = preserve original pixels)
    with edge-aware contour refinement and dilation buffers.
    """

    def generate_mask_from_items(
        self,
        image: Image.Image,
        items_to_remove: List[Dict[str, Any]],
        dilation_pixels: int = 14,
        feather_radius: int = 4
    ) -> Image.Image:
        """
        Builds a combined binary mask from a list of target items containing bounding boxes.
        
        Args:
            image: Original PIL Image.
            items_to_remove: List of item dictionaries with optional 'bbox' key [x1, y1, x2, y2].
            dilation_pixels: Amount to dilate mask to ensure complete object edge coverage.
            feather_radius: Gaussian blur feathering for seamless diffusion blending.
            
        Returns:
            PIL.Image in 'L' mode (single channel binary mask).
        """
        w, h = image.size
        mask_np = np.zeros((h, w), dtype=np.uint8)

        if not items_to_remove:
            # Default to lower-central furniture seating region if none provided
            y1, y2 = int(h * 0.45), int(h * 0.90)
            x1, x2 = int(w * 0.15), int(w * 0.85)
            cv2.rectangle(mask_np, (x1, y1), (x2, y2), 255, -1)
        else:
            for item in items_to_remove:
                bbox = item.get("bbox")
                if bbox and len(bbox) == 4:
                    # Check if normalized coordinates (0.0 to 1.0)
                    if all(0.0 <= v <= 1.0 for v in bbox):
                        # standard format [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max]
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            bx1, by1, bx2, by2 = bbox
                        else:
                            by1, bx1, by2, bx2 = bbox
                        
                        px1 = max(0, int(bx1 * w))
                        py1 = max(0, int(by1 * h))
                        px2 = min(w, int(bx2 * w))
                        py2 = min(h, int(by2 * h))
                    else:
                        px1, py1, px2, py2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    # Apply GrabCut / contour refinement within ROI
                    refined_roi = self._refine_object_mask(np.array(image), px1, py1, px2, py2)
                    mask_np[py1:py2, px1:px2] = np.maximum(mask_np[py1:py2, px1:px2], refined_roi)

        # 2. Dilate mask to engulf object drop shadows & perimeter fringes
        if dilation_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1))
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)

        mask_pil = Image.fromarray(mask_np, mode="L")

        # 3. Soft Gaussian feathering at boundaries
        if feather_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather_radius))

        return mask_pil

    def _refine_object_mask(
        self,
        img_np: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> np.ndarray:
        """
        Uses OpenCV GrabCut / edge thresholding within bounding box to obtain realistic object contour.
        """
        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w <= 10 or roi_h <= 10:
            return np.full((max(1, roi_h), max(1, roi_w)), 255, dtype=np.uint8)

        roi = img_np[y1:y2, x1:x2]
        
        # Try grabcut
        try:
            mask = np.zeros((roi_h, roi_w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            rect = (2, 2, roi_w - 4, roi_h - 4)
            cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            
            # 1 and 3 are foreground / probable foreground
            binary_roi = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
            
            # If grabcut removed too much (<25%), fallback to filled rectangle
            if np.count_nonzero(binary_roi) < (roi_w * roi_h * 0.25):
                return np.full((roi_h, roi_w), 255, dtype=np.uint8)
                
            return binary_roi
        except Exception:
            return np.full((roi_h, roi_w), 255, dtype=np.uint8)

    def generate_mask_overlay_preview(
        self,
        original_image: Image.Image,
        mask: Image.Image,
        overlay_color: Tuple[int, int, int] = (239, 68, 68),  # Crimson Red
        alpha: float = 0.50
    ) -> Image.Image:
        """
        Creates an architectural preview showing the masked inpaint regions highlighted in transparent color.
        """
        orig = original_image.convert("RGBA")
        m_np = np.array(mask.convert("L"))
        
        overlay = np.zeros((orig.height, orig.width, 4), dtype=np.uint8)
        mask_indices = m_np > 30
        
        overlay[mask_indices, 0] = overlay_color[0]
        overlay[mask_indices, 1] = overlay_color[1]
        overlay[mask_indices, 2] = overlay_color[2]
        overlay[mask_indices, 3] = (m_np[mask_indices] * alpha).astype(np.uint8)
        
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        return Image.alpha_composite(orig, overlay_img).convert("RGB")


# Singleton instance
segmenter_service = RoomSegmenter()
