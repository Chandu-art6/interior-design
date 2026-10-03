"""
YOLOv8 Room Analysis & Furniture Detection Service
Detects furniture items, calculates bounding boxes, confidence scores,
and computes spatial layout metrics (coverage, clutter, balance).
"""
import logging
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from config import settings

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


# Category mapping for indoor objects and furniture
FURNITURE_CATEGORY_MAP = {
    "chair": "Seating",
    "couch": "Seating",
    "sofa": "Seating",
    "bed": "Sleeping",
    "dining table": "Surface",
    "table": "Surface",
    "desk": "Surface",
    "tv": "Electronics",
    "laptop": "Electronics",
    "refrigerator": "Appliance",
    "oven": "Appliance",
    "sink": "Fixtures",
    "toilet": "Fixtures",
    "potted plant": "Decor",
    "vase": "Decor",
    "book": "Decor",
    "clock": "Decor",
    "wardrobe": "Storage",
    "cabinet": "Storage",
    "shelf": "Storage",
    "lamp": "Lighting"
}

# Category colors for UI annotation (RGB)
CATEGORY_COLORS = {
    "Seating": (99, 102, 241),     # Indigo
    "Surface": (59, 130, 246),     # Blue
    "Sleeping": (168, 85, 247),    # Purple
    "Storage": (234, 179, 8),      # Amber
    "Decor": (16, 185, 129),       # Emerald
    "Electronics": (239, 68, 68),  # Red
    "Appliance": (249, 115, 22),   # Orange
    "Fixtures": (20, 184, 166),    # Teal
    "Lighting": (234, 88, 12),     # Amber-Red
    "General": (148, 163, 184)     # Slate
}


class YOLOService:
    """Ultralytics YOLOv8 wrapper for room element and furniture detection."""

    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._is_loaded = False

    def load_model(self):
        """Loads or lazy-initializes the YOLOv8 model."""
        if self._is_loaded and self.model is not None:
            return self.model

        if not YOLO_AVAILABLE:
            logger.warning("Ultralytics package is not installed. Running in heuristic/fallback detection mode.")
            return None

        try:
            logger.info(f"Loading YOLO model: {self.model_name}...")
            # Use standard YOLO model or download automatically
            self.model = YOLO(self.model_name)
            self._is_loaded = True
            logger.info("YOLO model loaded successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_name}': {e}. Using fallback.")
            self.model = None
            return None

    def _calculate_spatial_metrics(
        self,
        detections: List[Dict[str, Any]],
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Computes spatial layout vectors:
        - Furniture coverage percentage
        - Furniture density
        - Clutter score (0.0 to 100.0)
        - Spatial horizontal balance (-1.0 to +1.0)
        """
        if not detections:
            return {
                "furniture_count": 0,
                "furniture_coverage_pct": 0.0,
                "furniture_density": 0.0,
                "spatial_clutter_score": 5.0,
                "spatial_balance": 0.0,
                "layout_distribution": "Spacious / Minimal"
            }

        total_image_area = float(img_width * img_height)
        
        # Calculate coverage using bounding box areas
        total_bbox_area = sum(d["area_pixels"] for d in detections)
        # Cap coverage at 100% to handle overlapping bounding boxes
        coverage_pct = min(100.0, round((total_bbox_area / total_image_area) * 100, 2))
        
        # Furniture density (items per normalized megapixel)
        density = round(len(detections) / (total_image_area / 1_000_000), 2)
        
        # Clutter score: combination of item count and area coverage
        count_factor = min(50.0, len(detections) * 6.0)
        coverage_factor = min(50.0, coverage_pct * 0.75)
        clutter_score = round(min(100.0, count_factor + coverage_factor), 1)

        # Spatial balance: average centroid X coordinate relative to center (0.5)
        avg_center_x = sum((d["bbox_normalized"][0] + d["bbox_normalized"][2]) / 2.0 for d in detections) / len(detections)
        spatial_balance = round((avg_center_x - 0.5) * 2.0, 2)  # -1.0 (left) to +1.0 (right)

        if clutter_score < 30.0:
            distribution_desc = "Spacious & Uncluttered"
        elif clutter_score < 60.0:
            distribution_desc = "Moderately Furnished"
        else:
            distribution_desc = "Dense / High Clutter"

        return {
            "furniture_count": len(detections),
            "furniture_coverage_pct": coverage_pct,
            "furniture_density": density,
            "spatial_clutter_score": clutter_score,
            "spatial_balance": spatial_balance,
            "layout_distribution": distribution_desc
        }

    def detect_furniture(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        conf_threshold: float = settings.YOLO_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Executes YOLO object detection on input room image.
        Returns detected furniture items and spatial layout analytics.
        """
        # Ensure image is PIL Image and get dimensions
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image_input.convert("RGB")

        img_w, img_h = pil_image.size
        model = self.load_model()

        detections: List[Dict[str, Any]] = []

        if model is not None:
            try:
                # Convert PIL to BGR numpy for YOLO inference
                img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                results = model.predict(source=img_cv, conf=conf_threshold, verbose=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names.get(cls_id, f"item_{cls_id}").lower()
                        conf = round(float(box.conf[0].item()), 3)
                        
                        # Coordinates: x1, y1, x2, y2
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        
                        # Normalized coordinates [0.0 - 1.0]
                        norm_box = [
                            round(max(0.0, min(1.0, x1 / img_w)), 4),
                            round(max(0.0, min(1.0, y1 / img_h)), 4),
                            round(max(0.0, min(1.0, x2 / img_w)), 4),
                            round(max(0.0, min(1.0, y2 / img_h)), 4)
                        ]
                        
                        area_px = max(0.0, (x2 - x1) * (y2 - y1))
                        area_ratio = round(area_px / (img_w * img_h), 4)
                        category = FURNITURE_CATEGORY_MAP.get(cls_name, "General")

                        detections.append({
                            "label": cls_name,
                            "category": category,
                            "confidence": conf,
                            "bbox_normalized": norm_box,
                            "bbox_pixels": [int(x1), int(y1), int(x2), int(y2)],
                            "area_pixels": area_px,
                            "area_ratio": area_ratio
                        })
            except Exception as e:
                logger.error(f"Error during YOLO prediction: {e}")

        # If no detections from YOLO (e.g. running offline / mock image / model unavailable), provide fallback
        if not detections and not YOLO_AVAILABLE:
            # Fallback mock detections for local testing
            mock_items = [
                {"label": "couch", "category": "Seating", "confidence": 0.91, "norm": [0.15, 0.45, 0.85, 0.88]},
                {"label": "coffee table", "category": "Surface", "confidence": 0.86, "norm": [0.35, 0.65, 0.65, 0.85]}
            ]
            for m in mock_items:
                nx1, ny1, nx2, ny2 = m["norm"]
                px1, py1, px2, py2 = int(nx1 * img_w), int(ny1 * img_h), int(nx2 * img_w), int(ny2 * img_h)
                area_px = (px2 - px1) * (py2 - py1)
                detections.append({
                    "label": m["label"],
                    "category": m["category"],
                    "confidence": m["confidence"],
                    "bbox_normalized": m["norm"],
                    "bbox_pixels": [px1, py1, px2, py2],
                    "area_pixels": area_px,
                    "area_ratio": round(area_px / (img_w * img_h), 4)
                })

        # Calculate layout metrics
        spatial_metrics = self._calculate_spatial_metrics(detections, img_w, img_h)

        return {
            "image_dimensions": {"width": img_w, "height": img_h},
            "detections_count": len(detections),
            "detections": detections,
            "spatial_metrics": spatial_metrics
        }

    def draw_detections(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        Draws visual bounding boxes, labels, and category badges onto the image.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            img = image_input.copy().convert("RGB")

        draw = ImageDraw.Draw(img)
        w, h = img.size

        for item in detections:
            norm_box = item.get("bbox_normalized")
            if not norm_box or len(norm_box) != 4:
                continue

            x1 = int(norm_box[0] * w)
            y1 = int(norm_box[1] * h)
            x2 = int(norm_box[2] * w)
            y2 = int(norm_box[3] * h)

            category = item.get("category", "General")
            color = CATEGORY_COLORS.get(category, (99, 102, 241))
            label = item.get("label", "Object").capitalize()
            conf = item.get("confidence", 0.0)

            # Draw outer bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label banner
            text = f"{label} ({int(conf * 100)}%)"
            text_bbox = draw.textbbox((x1, y1), text)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            banner_y1 = max(0, y1 - th - 8)
            banner_y2 = y1

            draw.rectangle([x1, banner_y1, x1 + tw + 10, banner_y2], fill=color)
            draw.text((x1 + 5, banner_y1 + 2), text, fill=(255, 255, 255))

        return img

    def draw_detections_overlay(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        detections: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> Image.Image:
        """
        Draws visual bounding boxes, labels, and category badges onto the image,
        and optionally saves the annotated image to output_path.
        """
        annotated_img = self.draw_detections(image_input, detections)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            annotated_img.save(str(output_path), quality=95)
        return annotated_img



# Singleton instance
yolo_service = YOLOService()
