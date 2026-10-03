"""
Verification Test for Targeted AI Room Redesign & Inpainting Pipeline
Tests Analyzer, Segmenter, Depth Extractor, ControlNet Inpaint Generator, and API Endpoints.
"""
import numpy as np
from PIL import Image
import os
import sys

from services.analyzer import analyzer_service
from services.segmenter import segmenter_service
from services.depth import depth_service
from services.generator import generator_service
from config import settings


def test_inpainting_pipeline():
    print("=" * 70)
    print("=== TESTING TARGETED INPAINTING & AI ROOM REDESIGN PIPELINE ===")
    print("=" * 70)

    # 1. Create a synthetic test room image
    w, h = 1024, 1024
    test_img_np = np.full((h, w, 3), 235, dtype=np.uint8)
    test_img_np[600:, :] = [215, 185, 145]  # Wood floor
    test_img_np[520:800, 200:800] = [60, 70, 90]  # Blue sofa
    test_img_np[700:860, 380:640] = [130, 70, 25]  # Coffee table
    test_img_np[150:450, 100:300] = [180, 210, 240]  # Window
    test_img = Image.fromarray(test_img_np)

    print("\n[Step 1] Testing Vision LLM Analyzer (services/analyzer.py)...")
    analysis = analyzer_service.analyze_room_for_inpainting(
        image=test_img,
        room_type="Living Room",
        target_style="Japandi",
        budget_tier="Mid-Range Premium",
        budget_limit=18000.0
    )
    print(f"  * Items to Remove: {len(analysis['items_to_remove'])}")
    for item in analysis["items_to_remove"]:
        print(f"    - [REMOVE] {item['name']}: {item['reason']}")
    
    print(f"  * Items to Keep: {len(analysis['items_to_keep'])}")
    for item in analysis["items_to_keep"]:
        print(f"    - [KEEP] {item['name']}: {item['reason']}")

    print(f"  * Purchase Suggestions: {len(analysis['purchase_suggestions'])}")
    for prod in analysis["purchase_suggestions"]:
        print(f"    - [SUGGESTION] {prod['name']} ({prod['estimated_cost']}) - {prod['placement_rationale']}")

    print(f"  * Inpainting Prompt: {analysis['inpainting_prompt'][:100]}...")
    assert len(analysis["items_to_remove"]) > 0, "Analyzer should identify items to remove"
    assert len(analysis["items_to_keep"]) > 0, "Analyzer should identify items to keep"
    assert len(analysis["purchase_suggestions"]) > 0, "Analyzer should provide purchase suggestions"
    assert len(analysis["inpainting_prompt"]) > 10, "Analyzer should generate inpainting prompt"
    print("  [OK] Analyzer passed.")

    print("\n[Step 2] Testing Precision Binary Mask Generator (services/segmenter.py)...")
    mask = segmenter_service.generate_mask_from_items(
        image=test_img,
        items_to_remove=analysis["items_to_remove"]
    )
    mask_np = np.array(mask)
    print(f"  * Mask Size: {mask.size} | Mode: {mask.mode}")
    print(f"  * Mask Min/Max: {mask_np.min()} / {mask_np.max()} | Non-zero Pixels: {np.count_nonzero(mask_np)}")
    assert mask.size == (1024, 1024), "Mask size must match input image"
    assert mask_np.max() > 0, "Mask must contain active inpainting region"
    print("  [OK] Segmenter passed.")

    print("\n[Step 3] Testing 3D Depth Map Extractor (services/depth.py)...")
    depth_map = depth_service.extract_depth(test_img)
    depth_np = np.array(depth_map)
    print(f"  * Depth Map Size: {depth_map.size} | Mode: {depth_map.mode}")
    print(f"  * Depth Range: {depth_np.min()} to {depth_np.max()}")
    assert depth_map.size == (1024, 1024), "Depth map size must match image"
    assert depth_np.max() > depth_np.min(), "Depth map must have non-zero dynamic range"
    print("  [OK] Depth Extractor passed.")

    print("\n[Step 4] Testing ControlNet Inpainting Generator (services/generator.py)...")
    inpaint_res = generator_service.inpaint(
        image=test_img,
        mask=mask,
        depth_map=depth_map,
        prompt=analysis["inpainting_prompt"],
        style="Japandi",
        num_variations=4
    )
    print(f"  * Primary Image: {inpaint_res['primary_image'].size}")
    print(f"  * Variations Generated: {len(inpaint_res['variations'])}")
    for var in inpaint_res["variations"]:
        print(f"    - [{var['variation_id']}] {var['title']} (Seed: {var['seed']})")

    # Verify that unmasked pixels are 100% identical
    unmasked_indices = mask_np == 0
    orig_unmasked = test_img_np[unmasked_indices]
    primary_np = np.array(inpaint_res["primary_image"])
    result_unmasked = primary_np[unmasked_indices]
    pixel_diff = np.abs(orig_unmasked.astype(int) - result_unmasked.astype(int)).max()
    print(f"  * Max Pixel Difference in Unmasked Regions: {pixel_diff} (100% preservation target: 0)")
    assert pixel_diff == 0, "Pixels outside the mask MUST be 100% preserved!"
    print("  [OK] ControlNet Inpainting & Exact Structure Preservation passed.")

    print("\n" + "=" * 70)
    print("[ALL PASSED] Targeted Inpainting Backend Pipeline verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_inpainting_pipeline()
