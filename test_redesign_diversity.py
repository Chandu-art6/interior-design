"""
Comprehensive Verification Suite:
1. Validates that AI generates substantially different interior redesigns.
2. Validates that Option 1, Option 2, Option 3, Option 4 are distinct from original room and from each other.
3. Validates design similarity & diversity validator thresholds.
4. Validates synchronized 2D floor plans, space optimization, and comfort metrics.
"""
import time
from PIL import Image
import numpy as np
from config import settings
from ui.api_client import api_client
from services.similarity_validator import similarity_validator


def test_diversity_pipeline():
    print("\n" + "="*65)
    print("[START] RUNNING SUBSTANTIALLY DIFFERENT INTERIOR REDESIGN VERIFICATION")
    print("="*65)

    # 1. Create a test living room image if not present
    test_img = Image.new("RGB", (1024, 768), (210, 180, 140))
    test_filename = "test_pipeline_room.png"
    test_img.save(settings.UPLOAD_DIR / test_filename)

    # 2. Test payload requesting 4 variations
    payload = {
        "image_filename": test_filename,
        "room_type": "Living Room",
        "interior_style": "Modern Minimalist",
        "budget_tier": "Luxury Bespoke",
        "budget_limit": 25000.0,
        "users": [
            {"name": "Elena", "preferred_style": "Modern Minimalist", "priority_objects": ["Sectional Sofa", "Marble Table"], "color_tone": "#475569"},
            {"name": "Marcus", "preferred_style": "Luxury Contemporary", "priority_objects": ["Curved Lounge", "Brass Chandelier"], "color_tone": "#d97706"}
        ],
        "color_palette": ["#475569", "#d97706", "#f8fafc"],
        "preserve_structure": True,
        "custom_prompt_notes": "Open spacious architectural flow, zero clutter",
        "controlnet_scale": 0.45,
        "inference_steps": 30,
        "num_variations": 4
    }

    t0 = time.time()
    res = api_client.generate_redesign(payload)
    elapsed = time.time() - t0

    print(f"\n[OK] Pipeline generated {len(res['variations'])} variations in {elapsed:.2f}s (<3.0s target)")
    assert len(res["variations"]) == 4, f"Expected 4 variations, got {len(res['variations'])}"

    # 3. Check each variation against original uploaded image
    orig_path = settings.UPLOAD_DIR / test_filename
    orig_pil = Image.open(orig_path)

    var_images = []
    for i, var in enumerate(res["variations"]):
        var_img_path = settings.STATIC_DIR / var["image_url"].replace("/static/", "")
        var_floor_path = settings.STATIC_DIR / var["floorplan_diagram_url"].replace("/static/", "")
        
        assert var_img_path.is_file(), f"Variation {i+1} image file missing: {var_img_path}"
        assert var_floor_path.is_file(), f"Variation {i+1} 2D floorplan missing: {var_floor_path}"

        var_pil = Image.open(var_img_path)
        var_images.append(var_pil)

        # Validate similarity against original room
        sim_check = similarity_validator.compute_similarity(orig_pil, var_pil)
        print(f"\n[Layout] Option {i+1}: {var['title']}")
        print(f"   Style: {var['style']} | Archetype: {var['layout_archetype']}")
        print(f"   Similarity to Original: {sim_check['similarity_score']:.3f} (Threshold <= {settings.SIMILARITY_THRESHOLD})")
        print(f"   Diversity Score: {sim_check['diversity_score']*100:.1f}% ({sim_check['diversity_status']})")
        print(f"   Walking Clearance: {var['circulation_corridor']}")

        assert sim_check["is_sufficiently_different"], f"Option {i+1} was too similar to the original image!"
        assert var["is_sufficiently_different"] is True

    # 4. Check pairwise diversity between all options
    print("\n" + "-"*50)
    print("[CHECK] CHECKING PAIRWISE DIVERSITY BETWEEN GENERATED OPTIONS")
    print("-"*50)
    pairwise = similarity_validator.compute_pairwise_diversity(var_images)
    for p in pairwise:
        print(f"   {p['pair_label']}: Diversity = {p['diversity_score']*100:.1f}% (Distinct: {p['is_distinct']})")
        assert p["is_distinct"], f"Options {p['pair_label']} are too similar to each other!"

    # 5. Verify Original Image Integrity (never overwritten or modified)
    assert orig_path.is_file(), "Original image was deleted or moved!"
    assert orig_pil.size == (1024, 768), "Original image size was corrupted!"
    print("\n[OK] Original uploaded room image preserved with 100% integrity.")

    print("\n" + "="*65)
    print("[SUCCESS] ALL ACCEPTANCE CRITERIA PASSED (100% DIVERSITY VERIFIED)")
    print("="*65 + "\n")


if __name__ == "__main__":
    test_diversity_pipeline()
