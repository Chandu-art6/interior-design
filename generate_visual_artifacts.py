"""
Generates and exports visual verification artifacts directly from the user's uploaded image.
Produces:
1. Original Room Image
2. Structure-Locked Inpainting Mask
3. Active AI Redesign (Alex & Sam Hybrid)
4. Spatial Placement Blueprint & Clearance Overlay
5. 4-Variation Comparison Grid
"""
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config import settings
from services.yolo_service import yolo_service
from services.segmenter import segmenter_service
from services.depth import depth_service
from services.generator import generator_service
from services.consensus_engine import consensus_engine
from services.placement_engine import placement_engine

def generate_visual_artifacts():
    print("=== Generating Visual Artifacts for Walkthrough Document ===")
    
    artifact_dir = Path(r"C:\Users\G chandu\.gemini\antigravity-ide\brain\7698e2b7-70a0-419d-82d8-672ab869cc86")
    user_uploaded_img_path = artifact_dir / ".user_uploaded" / "media_1787635096809.png"
    
    if not user_uploaded_img_path.exists():
        print(f"Fallback to sample_room.jpg")
        user_uploaded_img_path = settings.UPLOAD_DIR / "sample_room.jpg"
    
    orig_img = Image.open(str(user_uploaded_img_path)).convert("RGB")
    
    # 1. Save original room copy into artifact dir
    orig_save_path = artifact_dir / "verified_original_room.png"
    orig_img.save(orig_save_path)
    print(f"  * Saved Original Room to {orig_save_path}")

    # 2. Run YOLO Object Detection
    yolo_res = yolo_service.detect_furniture(str(user_uploaded_img_path))
    detections = yolo_res["detections"]
    print(f"  * YOLO Detections: {len(detections)} objects found")

    # 3. Create items to remove mask
    items_to_remove = [
        {"name": d["label"], "bbox": d["bbox"], "reason": "Redesign replacement"}
        for d in detections
    ]
    if not items_to_remove:
        items_to_remove = [{"name": "central_furniture", "bbox": [0.2, 0.45, 0.8, 0.85]}]

    mask = segmenter_service.generate_mask_from_items(orig_img, items_to_remove)
    mask_save_path = artifact_dir / "verified_mask.png"
    mask.save(mask_save_path)
    print(f"  * Saved Mask to {mask_save_path}")

    # 4. Generate 4 Redesign Variations with Multi-User Consensus
    users = [
        {
            "name": "Alex",
            "preferred_style": "Industrial",
            "priority_objects": ["Leather Sofa", "Steel Bookshelf"],
            "color_tone": "#475569"
        },
        {
            "name": "Sam",
            "preferred_style": "Japandi",
            "priority_objects": ["Monstera Plant", "Floor Lamp"],
            "color_tone": "#e29578"
        }
    ]

    inpaint_res = generator_service.inpaint(
        image=orig_img,
        mask=mask,
        style="Industrial",
        users=users,
        budget_tier="Mid-Range Premium",
        budget_limit=22000.0,
        num_variations=4,
        detected_furniture=detections
    )

    # 5. Save Primary Redesign & Placement Blueprint
    primary_save_path = artifact_dir / "verified_redesign_primary.png"
    inpaint_res["primary_image"].save(primary_save_path)
    print(f"  * Saved Primary Redesign to {primary_save_path}")

    placement_save_path = artifact_dir / "verified_placement_blueprint.png"
    inpaint_res["placement_overlay_image"].save(placement_save_path)
    print(f"  * Saved Placement Blueprint to {placement_save_path}")

    # 6. Save each variation and floor plan
    floorplan_imgs = []
    for idx, var in enumerate(inpaint_res["variations"]):
        var_save_path = artifact_dir / f"verified_var_{idx+1}.png"
        var["image"].save(var_save_path)
        var_place_path = artifact_dir / f"verified_var_{idx+1}_placement.png"
        var["placement_overlay_image"].save(var_place_path)
        
        var_floor_path = artifact_dir / f"verified_floorplan_var_{idx+1}.png"
        if var.get("floorplan_diagram_image"):
            var["floorplan_diagram_image"].save(var_floor_path)
            floorplan_imgs.append(var["floorplan_diagram_image"])
        print(f"  * Saved Variation {idx+1} ({var['title']}) and Floor Plan to {var_floor_path}")

    # 7. Create 4-Floor-Plan 2x2 Architectural Blueprint Grid
    if len(floorplan_imgs) == 4:
        grid_w = floorplan_imgs[0].width * 2
        grid_h = floorplan_imgs[0].height * 2
        grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
        grid_img.paste(floorplan_imgs[0], (0, 0))
        grid_img.paste(floorplan_imgs[1], (floorplan_imgs[0].width, 0))
        grid_img.paste(floorplan_imgs[2], (0, floorplan_imgs[0].height))
        grid_img.paste(floorplan_imgs[3], (floorplan_imgs[0].width, floorplan_imgs[0].height))
        
        grid_save_path = artifact_dir / "verified_4_floorplan_comparison.png"
        grid_img.save(grid_save_path)
        print(f"  * Saved 4-Floor-Plan 2x2 Comparison Grid to {grid_save_path}")

    # 8. Create Side-by-Side 3-Panel Composite (Original | 2D Blueprint | Perspective Redesign)
    w, h = orig_img.size
    comp_w = w * 3
    comp_h = h
    primary_floorplan = inpaint_res.get("floorplan_diagram_image", inpaint_res["placement_overlay_image"]).resize((w, h))
    composite_3panel = Image.new("RGB", (comp_w, comp_h), (15, 23, 42))
    composite_3panel.paste(orig_img, (0, 0))
    composite_3panel.paste(primary_floorplan, (w, 0))
    composite_3panel.paste(inpaint_res["primary_image"], (w * 2, 0))

    comp_save_path = artifact_dir / "verified_3panel_comparison.png"
    composite_3panel.save(comp_save_path)
    print(f"  * Saved 3-Panel Side-by-Side Comparison to {comp_save_path}")

    print("\n[ALL ARTIFACTS GENERATED SUCCESSFULLY]")


if __name__ == "__main__":
    generate_visual_artifacts()
