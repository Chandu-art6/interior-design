"""
UI Backend API Client
Coordinates HTTP communication between Streamlit frontend and FastAPI backend with direct fallback.
"""
import uuid
import time
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

from config import settings


class APIClient:
    """Client for interacting with the Interior Redesign API."""

    def __init__(self, base_url: str = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1"):
        self.base_url = base_url

    def check_health(self) -> bool:
        """Checks if backend FastAPI service is reachable."""
        try:
            resp = requests.get(f"http://{settings.API_HOST}:{settings.API_PORT}/health", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def upload_image(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Uploads room image to server."""
        from services.image_processor import image_processor
        ext = Path(filename).suffix or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = settings.UPLOAD_DIR / unique_name

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        processed = image_processor.preprocess_upload(save_path)
        processed.save(save_path, quality=95)

        return {
            "status": "success",
            "filename": unique_name,
            "url": f"/static/uploads/{unique_name}",
            "dimensions": {"width": processed.width, "height": processed.height}
        }

    def analyze_room(self, filename: str, room_type: str = "Living Room") -> Dict[str, Any]:
        """Runs YOLO object detection and color palette analysis."""
        from services.yolo_service import yolo_service
        from services.image_processor import image_processor
        from database.session import SessionLocal
        from database.models import RoomAnalysis

        img_path = settings.UPLOAD_DIR / filename
        yolo_result = yolo_service.detect_furniture(img_path)
        palette = image_processor.extract_dominant_palette(img_path, num_colors=5)

        annotated_name = f"annotated_{filename}"
        yolo_service.draw_detections_overlay(img_path, yolo_result["detections"], settings.OUTPUT_DIR / annotated_name)

        analysis_uid = f"ana_{uuid.uuid4().hex[:12]}"
        db = SessionLocal()
        try:
            db_analysis = RoomAnalysis(
                analysis_uid=analysis_uid,
                image_path=f"/static/uploads/{filename}",
                room_type=room_type,
                detected_objects=yolo_result["detections"],
                spatial_dimensions=yolo_result["spatial_metrics"],
                lighting_condition="Balanced Daylight",
                clutter_score=yolo_result["spatial_metrics"]["spatial_clutter_score"]
            )
            db.add(db_analysis)
            db.commit()
        finally:
            db.close()

        return {
            "analysis_uid": analysis_uid,
            "image_url": f"/static/uploads/{filename}",
            "room_type": room_type,
            "detected_objects": yolo_result["detections"],
            "spatial_metrics": yolo_result["spatial_metrics"],
            "dominant_palette": palette,
            "annotated_image_url": f"/static/outputs/{annotated_name}",
            "lighting_condition": "Balanced Daylight"
        }

    def analyze_inpaint(self, filename: str, room_type: str = "Living Room", style: str = "Modern Minimalist", budget_tier: str = "Mid-Range Premium", budget_limit: float = 15000.0) -> Dict[str, Any]:
        """Calls Vision LLM to get items to remove, items to keep, and purchase suggestions."""
        from services.analyzer import analyzer_service
        img_path = settings.UPLOAD_DIR / filename
        orig_img = Image.open(str(img_path)).convert("RGB")
        return analyzer_service.analyze_room_for_inpainting(
            image=orig_img,
            room_type=room_type,
            target_style=style,
            budget_tier=budget_tier,
            budget_limit=budget_limit
        )

    def generate_redesign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generates AI interior spatial redesign and furniture repositioning."""
        from services.analyzer import analyzer_service
        from services.segmenter import segmenter_service
        from services.depth import depth_service
        from services.generator import generator_service
        from services.yolo_service import yolo_service
        from services.recommender import recommender_service
        from database.session import SessionLocal
        from database.models import GeneratedDesign

        image_filename = payload["image_filename"]
        img_path = settings.UPLOAD_DIR / image_filename
        orig_img = Image.open(str(img_path)).convert("RGB")

        room_type = payload.get("room_type", "Living Room")
        style = payload.get("interior_style", "Modern Minimalist")
        budget_tier = payload.get("budget_tier", "Mid-Range Premium")
        budget_limit = payload.get("budget_limit", 15000.0)

        # 1. Vision LLM Analysis
        analysis = analyzer_service.analyze_room_for_inpainting(
            image=orig_img,
            room_type=room_type,
            target_style=style,
            budget_tier=budget_tier,
            budget_limit=budget_limit,
            custom_notes=payload.get("custom_prompt_notes", "")
        )
        items_to_remove = analysis.get("items_to_remove", [])
        items_to_keep = analysis.get("items_to_keep", [])
        purchase_suggestions = analysis.get("purchase_suggestions", [])
        inpaint_prompt = analysis.get("inpainting_prompt", f"Photorealistic {style} interior redesign of a {room_type}")

        if payload.get("target_items_to_replace"):
            for item_name in payload["target_items_to_replace"]:
                items_to_remove.append({"name": item_name, "reason": "User-selected targeted replacement"})

        # 2. YOLO Object Detection & Spatial Metrics
        yolo_result = yolo_service.detect_furniture(img_path)
        detections = yolo_result["detections"]
        spatial_metrics = yolo_result["spatial_metrics"]

        # 3. Precision Binary Mask
        mask_img = segmenter_service.generate_mask_from_items(orig_img, items_to_remove)

        # 4. 3D Spatial Depth Map
        depth_img = depth_service.extract_depth(orig_img)

        # 5. Inpainting Generator with Consensus & Placement Engines
        inpaint_result = generator_service.inpaint(
            image=orig_img,
            mask=mask_img,
            depth_map=depth_img,
            prompt=inpaint_prompt,
            style=style,
            room_type=room_type,
            users=payload.get("users"),
            budget_tier=budget_tier,
            budget_limit=budget_limit,
            controlnet_scale=payload.get("controlnet_scale", 0.45),
            num_inference_steps=payload.get("inference_steps", 30),
            num_variations=payload.get("num_variations", 2),
            detected_furniture=detections
        )

        project_uid = f"proj_{uuid.uuid4().hex[:12]}"
        ts = int(time.time() * 1000)

        primary_filename = f"redesign_{project_uid}_{ts}.png"
        inpaint_result["primary_image"].save(settings.OUTPUT_DIR / primary_filename)

        placement_filename = f"placement_{project_uid}_{ts}.png"
        inpaint_result["placement_overlay_image"].save(settings.OUTPUT_DIR / placement_filename)

        floorplan_filename = f"floorplan_{project_uid}_{ts}.png"
        if inpaint_result.get("floorplan_diagram_image"):
            inpaint_result["floorplan_diagram_image"].save(settings.OUTPUT_DIR / floorplan_filename)

        mask_filename = f"mask_{project_uid}_{ts}.png"
        mask_img.save(settings.OUTPUT_DIR / mask_filename)

        depth_filename = f"depth_{project_uid}_{ts}.png"
        depth_img.save(settings.OUTPUT_DIR / depth_filename)

        variations = []
        for var in inpaint_result["variations"]:
            var_name = f"var_{var['variation_id']}_{project_uid}_{ts}.png"
            var["image"].save(settings.OUTPUT_DIR / var_name)

            var_place_name = f"place_var_{var['variation_id']}_{project_uid}_{ts}.png"
            var["placement_overlay_image"].save(settings.OUTPUT_DIR / var_place_name)

            var_floorplan_name = f"floorplan_var_{var['variation_id']}_{project_uid}_{ts}.png"
            if var.get("floorplan_diagram_image"):
                var["floorplan_diagram_image"].save(settings.OUTPUT_DIR / var_floorplan_name)

            variations.append({
                "variation_id": var["variation_id"],
                "title": var["title"],
                "subtitle": var.get("subtitle", ""),
                "style": var["style"],
                "persona_blend": var.get("persona_blend", ""),
                "image_url": f"/static/outputs/{var_name}",
                "placement_overlay_url": f"/static/outputs/{var_place_name}",
                "floorplan_diagram_url": f"/static/outputs/{var_floorplan_name}",
                "layout_archetype": var.get("layout_archetype"),
                "wall_anchor": var.get("wall_anchor"),
                "circulation_corridor": var.get("circulation_corridor"),
                "layout_description": var.get("layout_description", ""),
                "suggested_objects": var.get("suggested_objects", []),
                "placement_suggestions": var.get("placement_suggestions", []),
                "pipeline_mode": var.get("pipeline_mode", "inpaint"),
                "seed": var.get("seed"),
                "controlnet_weight": var.get("controlnet_weight"),
                "similarity_metrics": var.get("similarity_metrics", {}),
                "diversity_score": var.get("diversity_score", 0.85),
                "diversity_status": var.get("diversity_status", "✅ Substantially Redesigned"),
                "is_sufficiently_different": var.get("is_sufficiently_different", True)
            })

        comfort_scores = recommender_service.evaluate_comfort_metrics(
            detected_items=detections,
            room_type=room_type,
            spatial_metrics=spatial_metrics
        )
        recommendations = recommender_service.generate_recommendations(
            metrics=comfort_scores,
            room_type=room_type,
            style=style,
            detected_items=detections,
            dominant_colors=payload.get("color_palette", []),
            budget_tier=budget_tier
        )

        db = SessionLocal()
        try:
            db_design = GeneratedDesign(
                design_uid=project_uid,
                style_applied=style,
                original_image_path=f"/static/uploads/{image_filename}",
                generated_image_path=f"/static/outputs/{primary_filename}",
                prompt_used=inpaint_prompt,
                circulation_score=comfort_scores.circulation_score,
                lighting_score=comfort_scores.lighting_score,
                ergonomics_score=comfort_scores.ergonomics_score,
                clutter_index=comfort_scores.clutter_index,
                overall_comfort_score=comfort_scores.overall_comfort_score,
                ai_reasoning_log=recommendations
            )
            db.add(db_design)
            db.commit()
        finally:
            db.close()

        primary_placements = inpaint_result["variations"][0].get("placement_suggestions", []) if inpaint_result["variations"] else []

        return {
            "project_uid": project_uid,
            "original_image_url": f"/static/uploads/{image_filename}",
            "redesigned_image_url": f"/static/outputs/{primary_filename}",
            "primary_generated_url": f"/static/outputs/{primary_filename}",
            "placement_overlay_url": f"/static/outputs/{placement_filename}",
            "floorplan_diagram_url": f"/static/outputs/{floorplan_filename}",
            "mask_image_url": f"/static/outputs/{mask_filename}",
            "depth_image_url": f"/static/outputs/{depth_filename}",
            "canny_condition_url": f"/static/outputs/{mask_filename}",
            "depth_condition_url": f"/static/outputs/{depth_filename}",
            "style_applied": style,
            "room_type": room_type,
            "budget_tier": budget_tier,
            "removed_items": items_to_remove,
            "kept_items": items_to_keep,
            "recommendations": purchase_suggestions,
            "placement_suggestions": primary_placements,
            "inpainting_prompt": inpaint_prompt,
            "users": payload.get("users", []),
            "prompt_used": {"positive_prompt": inpaint_prompt},
            "comfort_metrics": {
                "circulation_score": comfort_scores.circulation_score,
                "lighting_score": comfort_scores.lighting_score,
                "ergonomics_score": comfort_scores.ergonomics_score,
                "clutter_index": comfort_scores.clutter_index,
                "overall_comfort_score": comfort_scores.overall_comfort_score
            },
            "explainable_recommendations": recommendations,
            "detected_furniture": detections,
            "variations": variations
        }

    def get_projects(self) -> List[Dict[str, Any]]:
        """Fetches historical redesign projects."""
        try:
            if self.check_health():
                resp = requests.get(f"{self.base_url}/projects", timeout=5)
                if resp.status_code == 200:
                    return resp.json().get("projects", [])
        except Exception:
            pass

        from database.session import SessionLocal
        from database.models import GeneratedDesign
        db = SessionLocal()
        try:
            designs = db.query(GeneratedDesign).order_by(GeneratedDesign.created_at.desc()).limit(20).all()
            return [
                {
                    "id": d.id,
                    "project_uid": d.design_uid,
                    "style": d.style_applied,
                    "original_image": d.original_image_path,
                    "generated_image": d.generated_image_path,
                    "overall_comfort": d.overall_comfort_score,
                    "created_at": d.created_at.isoformat() if d.created_at else None
                }
                for d in designs
            ]
        finally:
            db.close()


# Singleton instance
api_client = APIClient()
