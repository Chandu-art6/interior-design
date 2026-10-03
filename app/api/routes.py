import os
import time
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from config import settings
from database.session import get_db
from database.models import User, Preference, RoomAnalysis, GeneratedDesign
from services.image_processor import image_processor
from services.yolo_service import yolo_service
from services.controlnet_service import controlnet_service
from services.recommender import recommender_service
from services.analyzer import analyzer_service
from services.segmenter import segmenter_service
from services.depth import depth_service
from services.generator import generator_service

from app.schemas.design import (
    RoomAnalysisResponse,
    DesignRedesignRequest,
    DesignResponse,
    FurnitureDetectionItem,
    SpatialLayoutMetrics,
    DominantColorItem,
    DesignVariationItem,
    ComfortMetricScore,
    ItemChangeItem,
    PurchaseSuggestionItem,
    InpaintAnalysisResult
)

router = APIRouter(prefix="/api/v1", tags=["Interior Design Services"])


@router.post("/upload", summary="Upload room photograph for redesign pipeline")
async def upload_room_image(file: UploadFile = File(...)):
    """Accepts an image file, standardizes resolution, and saves to static uploads."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")

    ext = Path(file.filename).suffix or ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    raw_path = settings.UPLOAD_DIR / unique_filename

    # Save uploaded file
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Standardize image dimensions & color profiles
    processed_img = image_processor.preprocess_upload(raw_path)
    processed_img.save(raw_path, quality=95)

    return {
        "status": "success",
        "filename": unique_filename,
        "url": f"/static/uploads/{unique_filename}",
        "dimensions": {"width": processed_img.width, "height": processed_img.height}
    }


@router.post("/analyze", response_model=RoomAnalysisResponse, summary="Analyze room geometry, furniture, and lighting")
async def analyze_room(
    filename: str = Form(...),
    room_type: str = Form("Living Room"),
    db: Session = Depends(get_db)
):
    """
    Executes YOLOv8 furniture detection, layout spatial metrics,
    and k-means color palette extraction.
    """
    image_path = settings.UPLOAD_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in upload directory.")

    # 1. Object Detection & Spatial Metrics
    yolo_result = yolo_service.detect_furniture(image_path)
    detections = yolo_result["detections"]
    spatial_metrics = yolo_result["spatial_metrics"]

    # 2. Dominant Color Extraction
    dominant_palette = image_processor.extract_dominant_palette(image_path, num_colors=5)

    # 3. Save Annotated Bounding Box Visualizer
    annotated_filename = f"annotated_{filename}"
    annotated_path = settings.OUTPUT_DIR / annotated_filename
    yolo_service.draw_detections_overlay(image_path, detections, annotated_path)

    # 4. Store Analysis Record in DB
    analysis_uid = f"ana_{uuid.uuid4().hex[:12]}"
    db_analysis = RoomAnalysis(
        analysis_uid=analysis_uid,
        image_path=f"/static/uploads/{filename}",
        room_type=room_type,
        detected_objects=detections,
        spatial_dimensions=spatial_metrics,
        lighting_condition="Balanced Daylight",
        clutter_score=spatial_metrics["spatial_clutter_score"]
    )
    db.add(db_analysis)
    db.commit()

    detection_models = [FurnitureDetectionItem(**d) for d in detections]
    palette_items = [DominantColorItem(**c) for c in dominant_palette]
    spatial_model = SpatialLayoutMetrics(**spatial_metrics)

    return RoomAnalysisResponse(
        analysis_uid=analysis_uid,
        image_url=f"/static/uploads/{filename}",
        room_type=room_type,
        detected_objects=detection_models,
        spatial_metrics=spatial_model,
        dominant_palette=palette_items,
        annotated_image_url=f"/static/outputs/{annotated_filename}",
        lighting_condition="Balanced Daylight"
    )


@router.post("/analyze-inpaint", response_model=InpaintAnalysisResult, summary="Vision LLM room analysis for targeted inpainting")
async def analyze_inpaint(
    filename: str = Form(...),
    room_type: str = Form("Living Room"),
    interior_style: str = Form("Modern Minimalist"),
    budget_tier: str = Form("Mid-Range Premium"),
    budget_limit: float = Form(15000.0)
):
    """
    Vision LLM analysis extracting items to remove, items to keep, purchase suggestions,
    and the tailored inpainting prompt.
    """
    image_path = settings.UPLOAD_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in upload directory.")

    orig_img = Image.open(str(image_path)).convert("RGB")
    analysis = analyzer_service.analyze_room_for_inpainting(
        image=orig_img,
        room_type=room_type,
        target_style=interior_style,
        budget_tier=budget_tier,
        budget_limit=budget_limit
    )

    return InpaintAnalysisResult(
        items_to_remove=[ItemChangeItem(**i) for i in analysis.get("items_to_remove", [])],
        items_to_keep=[ItemChangeItem(**i) for i in analysis.get("items_to_keep", [])],
        purchase_suggestions=[PurchaseSuggestionItem(**p) for p in analysis.get("purchase_suggestions", [])],
        inpainting_prompt=analysis.get("inpainting_prompt", ""),
        style_rationale=analysis.get("style_rationale")
    )


@router.post("/redesign", response_model=DesignResponse, summary="Targeted inpainting redesign pipeline with ControlNet and depth preservation")
async def generate_redesign(
    request: DesignRedesignRequest,
    db: Session = Depends(get_db)
):
    """
    Executes the 4-step Targeted Inpainting Pipeline:
    1. Vision Model Analysis: Identifies items to remove, items to keep, and purchase suggestions.
    2. Precision Mask Generation: Creates a binary mask isolating only target objects.
    3. 3D Spatial Depth Extraction: Extracts a depth map to preserve room perspective.
    4. ControlNet Inpainting: Inpaints new style elements while locking untouched room structure.
    """
    image_path = settings.UPLOAD_DIR / request.image_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{request.image_filename}' not found in upload directory.")

    orig_img = Image.open(str(image_path)).convert("RGB")

    # --------------------------------------------------------------------------
    # Step 1: Vision Model Analysis (Items to remove, keep, and purchase suggestions)
    # --------------------------------------------------------------------------
    analysis = analyzer_service.analyze_room_for_inpainting(
        image=orig_img,
        room_type=request.room_type,
        target_style=request.interior_style,
        budget_tier=request.budget_tier,
        budget_limit=request.budget_limit or 15000.0,
        custom_notes=request.custom_prompt_notes
    )
    items_to_remove = analysis.get("items_to_remove", [])
    items_to_keep = analysis.get("items_to_keep", [])
    purchase_suggestions = analysis.get("purchase_suggestions", [])
    inpaint_prompt = analysis.get("inpainting_prompt", f"photorealistic {request.interior_style} room")

    # If user provided explicit target items to replace, append them
    if request.target_items_to_replace:
        for item_name in request.target_items_to_replace:
            items_to_remove.append({"name": item_name, "reason": "User-selected targeted replacement"})

    # --------------------------------------------------------------------------
    # Step 2: YOLO Object Detection & Spatial Metrics
    # --------------------------------------------------------------------------
    yolo_result = yolo_service.detect_furniture(image_path)
    detections = yolo_result["detections"]
    spatial_metrics = yolo_result["spatial_metrics"]

    # --------------------------------------------------------------------------
    # Step 3: Precision Binary Mask Generation (Isolates only the target items)
    # --------------------------------------------------------------------------
    mask_img = segmenter_service.generate_mask_from_items(
        image=orig_img,
        items_to_remove=items_to_remove,
        dilation_pixels=14,
        feather_radius=4
    )

    # --------------------------------------------------------------------------
    # Step 4: Geometry Preservation (3D Depth Map Extraction)
    # --------------------------------------------------------------------------
    depth_img = depth_service.extract_depth(orig_img)

    # --------------------------------------------------------------------------
    # Step 5: Multi-User Consensus & Structure-Preserving Inpainting Pipeline
    # --------------------------------------------------------------------------
    inpaint_result = generator_service.inpaint(
        image=orig_img,
        mask=mask_img,
        depth_map=depth_img,
        prompt=inpaint_prompt,
        style=request.interior_style,
        room_type=request.room_type,
        users=[u.dict() if hasattr(u, "dict") else u.model_dump() for u in request.users] if request.users else None,
        budget_tier=request.budget_tier,
        budget_limit=request.budget_limit or 15000.0,
        controlnet_scale=request.controlnet_scale,
        num_inference_steps=request.inference_steps,
        num_variations=request.num_variations,
        detected_furniture=detections
    )

    project_uid = f"proj_{uuid.uuid4().hex[:12]}"
    ts = int(time.time() * 1000)

    # Save Primary Redesign Image (Lossless PNG)
    primary_filename = f"redesign_{project_uid}_{ts}.png"
    primary_path = settings.OUTPUT_DIR / primary_filename
    inpaint_result["primary_image"].save(primary_path)

    # Save Primary Placement Blueprint Overlay
    placement_filename = f"placement_{project_uid}_{ts}.png"
    placement_path = settings.OUTPUT_DIR / placement_filename
    inpaint_result["placement_overlay_image"].save(placement_path)

    # Save Primary 2D Architectural Floor Plan Diagram
    floorplan_filename = f"floorplan_{project_uid}_{ts}.png"
    floorplan_path = settings.OUTPUT_DIR / floorplan_filename
    if inpaint_result.get("floorplan_diagram_image"):
        inpaint_result["floorplan_diagram_image"].save(floorplan_path)

    # Save Binary Mask & Depth Maps
    mask_filename = f"mask_{project_uid}_{ts}.png"
    mask_path = settings.OUTPUT_DIR / mask_filename
    mask_img.save(mask_path)

    depth_filename = f"depth_{project_uid}_{ts}.png"
    depth_path = settings.OUTPUT_DIR / depth_filename
    depth_img.save(depth_path)

    # Save Variations
    variation_models = []
    for var in inpaint_result["variations"]:
        var_filename = f"var_{var['variation_id']}_{project_uid}_{ts}.png"
        var_path = settings.OUTPUT_DIR / var_filename
        var["image"].save(var_path)

        var_placement_filename = f"place_var_{var['variation_id']}_{project_uid}_{ts}.png"
        var_place_path = settings.OUTPUT_DIR / var_placement_filename
        var["placement_overlay_image"].save(var_place_path)

        var_floorplan_filename = f"floorplan_var_{var['variation_id']}_{project_uid}_{ts}.png"
        var_floorplan_path = settings.OUTPUT_DIR / var_floorplan_filename
        if var.get("floorplan_diagram_image"):
            var["floorplan_diagram_image"].save(var_floorplan_path)

        variation_models.append(
            DesignVariationItem(
                variation_id=var["variation_id"],
                title=var["title"],
                subtitle=var.get("subtitle", ""),
                style=var["style"],
                persona_blend=var.get("persona_blend", ""),
                image_url=f"/static/outputs/{var_filename}",
                placement_overlay_url=f"/static/outputs/{var_placement_filename}",
                floorplan_diagram_url=f"/static/outputs/{var_floorplan_filename}",
                layout_archetype=var.get("layout_archetype"),
                wall_anchor=var.get("wall_anchor"),
                circulation_corridor=var.get("circulation_corridor"),
                layout_description=var.get("layout_description", ""),
                suggested_objects=var.get("suggested_objects", []),
                placement_suggestions=var.get("placement_suggestions", []),
                pipeline_mode=var.get("pipeline_mode", "inpaint"),
                seed=var.get("seed"),
                controlnet_weight=var.get("controlnet_weight")
            )
        )

    # --------------------------------------------------------------------------
    # Comfort Metrics & Explainable Recommendations
    # --------------------------------------------------------------------------
    comfort_scores = recommender_service.evaluate_comfort_metrics(
        detected_items=detections,
        room_type=request.room_type,
        spatial_metrics=spatial_metrics
    )

    recommendations = recommender_service.generate_recommendations(
        metrics=comfort_scores,
        room_type=request.room_type,
        style=request.interior_style,
        detected_items=detections,
        dominant_colors=request.color_palette,
        budget_tier=request.budget_tier
    )

    # --------------------------------------------------------------------------
    # Persist to Database
    # --------------------------------------------------------------------------
    db_design = GeneratedDesign(
        design_uid=project_uid,
        style_applied=request.interior_style,
        original_image_path=f"/static/uploads/{request.image_filename}",
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

    detection_models = [FurnitureDetectionItem(**d) for d in detections]
    removed_items_models = [ItemChangeItem(**i) for i in items_to_remove]
    kept_items_models = [ItemChangeItem(**i) for i in items_to_keep]
    purchase_models = [PurchaseSuggestionItem(**p) for p in purchase_suggestions]
    primary_placements = inpaint_result["variations"][0].get("placement_suggestions", []) if inpaint_result["variations"] else []

    return DesignResponse(
        project_uid=project_uid,
        original_image_url=f"/static/uploads/{request.image_filename}",
        redesigned_image_url=f"/static/outputs/{primary_filename}",
        primary_generated_url=f"/static/outputs/{primary_filename}",
        placement_overlay_url=f"/static/outputs/{placement_filename}",
        floorplan_diagram_url=f"/static/outputs/{floorplan_filename}",
        mask_image_url=f"/static/outputs/{mask_filename}",
        depth_image_url=f"/static/outputs/{depth_filename}",
        canny_condition_url=f"/static/outputs/{mask_filename}",
        depth_condition_url=f"/static/outputs/{depth_filename}",
        style_applied=request.interior_style,
        room_type=request.room_type,
        budget_tier=request.budget_tier,
        removed_items=removed_items_models,
        kept_items=kept_items_models,
        recommendations=purchase_models,
        placement_suggestions=primary_placements,
        inpainting_prompt=inpaint_prompt,
        users=request.users or [],
        prompt_used={"positive_prompt": inpaint_prompt},
        comfort_metrics=comfort_scores,
        explainable_recommendations=recommendations,
        detected_furniture=detection_models,
        variations=variation_models
    )


@router.get("/projects", summary="List historical redesign projects")
async def list_projects(db: Session = Depends(get_db)):
    """Fetches saved designs from SQLite database."""
    designs = db.query(GeneratedDesign).order_by(GeneratedDesign.created_at.desc()).limit(20).all()
    return {
        "count": len(designs),
        "projects": [
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
    }


@router.get("/projects/{project_uid}", summary="Get specific design project details")
async def get_project_detail(project_uid: str, db: Session = Depends(get_db)):
    """Retrieves full details, scores, and rationales for a specific design."""
    design = db.query(GeneratedDesign).filter(GeneratedDesign.design_uid == project_uid).first()
    if not design:
        raise HTTPException(status_code=404, detail=f"Project '{project_uid}' not found.")

    return {
        "project_uid": design.design_uid,
        "style_applied": design.style_applied,
        "original_image": design.original_image_path,
        "generated_image": design.generated_image_path,
        "prompt_used": design.prompt_used,
        "comfort_metrics": {
            "circulation_score": design.circulation_score,
            "lighting_score": design.lighting_score,
            "ergonomics_score": design.ergonomics_score,
            "clutter_index": design.clutter_index,
            "overall_comfort_score": design.overall_comfort_score
        },
        "ai_reasoning_log": design.ai_reasoning_log,
        "created_at": design.created_at.isoformat() if design.created_at else None
    }
