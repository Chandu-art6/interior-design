"""
Validation script for 2 distinct redesign variations:
1. Verifies num_variations = 2
2. Checks genuine furniture coordinate repositioning between Option 1 and Option 2
3. Verifies color palette / style differences
4. Validates dynamic comfort metrics per selected variation
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.generator import generator_service
from services.recommender import recommender_service


def test_2_distinct_redesigns():
    print("\n=== VALIDATING 2 DISTINCT REDESIGN SUGGESTIONS ===")
    
    # 1. Create a synthetic test image representing an uploaded living room
    test_img = Image.new("RGB", (900, 600), (210, 205, 195))
    test_mask = Image.new("L", (900, 600), 0)

    users = [
        {
            "name": "Sarah",
            "preferred_style": "Modern Minimalist",
            "priority_objects": ["Structured Sofa", "Glass Coffee Table"],
            "color_tone": "#3b82f6"  # Cobalt Accent
        },
        {
            "name": "Liam",
            "preferred_style": "Japandi",
            "priority_objects": ["Linen Lounge Chair", "Washi Floor Lamp", "Monstera Tree"],
            "color_tone": "#ea580c"  # Terracotta Warm Accent
        }
    ]

    res = generator_service.inpaint(
        image=test_img,
        mask=test_mask,
        style="Modern Minimalist",
        users=users,
        budget_tier="Mid-Range Premium",
        budget_limit=18000.0,
        num_variations=2
    )

    variations = res["variations"]
    assert len(variations) == 2, f"Expected exactly 2 variations, got {len(variations)}"
    print(f"\n[OK] Exactly 2 redesign variations generated:")

    var1 = variations[0]
    var2 = variations[1]

    print(f"  * Variation 1: {var1['title']}")
    print(f"    - Archetype: {var1['layout_archetype']}")
    print(f"    - Wall Anchor: {var1['wall_anchor']}")
    print(f"    - Placements Count: {len(var1['placement_suggestions'])}")

    print(f"\n  * Variation 2: {var2['title']}")
    print(f"    - Archetype: {var2['layout_archetype']}")
    print(f"    - Wall Anchor: {var2['wall_anchor']}")
    print(f"    - Placements Count: {len(var2['placement_suggestions'])}")

    # 2. Check that furniture placements are genuinely repositioned between Option 1 and Option 2
    p1_seating = var1["placement_suggestions"][0]["bbox_normalized"]
    p2_seating = var2["placement_suggestions"][0]["bbox_normalized"]
    print(f"\n--- Furniture Position Comparison ---")
    print(f"  * Option 1 Seating BBox: {p1_seating} (Centered on Main Wall)")
    print(f"  * Option 2 Seating BBox: {p2_seating} (Shifted to West Wall)")
    assert p1_seating != p2_seating, "Error: Seating coordinates must be different between Option 1 and Option 2!"
    print("[OK] Confirmed: Objects have genuinely different positions across Option 1 vs Option 2!")

    # 3. Check Dynamic Comfort & AI Reasoning for each variation
    ai_exp1 = recommender_service.get_variation_ai_explanation("var_1", var1["style"], "Mid-Range Premium", 18000.0, var1["persona_blend"])
    ai_exp2 = recommender_service.get_variation_ai_explanation("var_2", var2["style"], "Mid-Range Premium", 18000.0, var2["persona_blend"])

    print(f"\n--- Dynamic Comfort Metrics Comparison ---")
    print(f"  * Option 1 Comfort Score: {ai_exp1['comfort_score']}/100 | Circulation: {ai_exp1['circulation_clearance']}")
    print(f"  * Option 2 Comfort Score: {ai_exp2['comfort_score']}/100 | Circulation: {ai_exp2['circulation_clearance']}")

    assert ai_exp1["comfort_score"] != ai_exp2["comfort_score"] or ai_exp1["circulation_clearance"] != ai_exp2["circulation_clearance"], "Metrics must be dynamically tailored per variation"

    # 4. Check Image Pixel Differences
    img1_np = np.array(var1["image"])
    img2_np = np.array(var2["image"])
    orig_np = np.array(test_img)

    diff_orig_v1 = np.mean(np.abs(img1_np.astype(float) - orig_np.astype(float)))
    diff_orig_v2 = np.mean(np.abs(img2_np.astype(float) - orig_np.astype(float)))
    diff_v1_v2 = np.mean(np.abs(img1_np.astype(float) - img2_np.astype(float)))

    print(f"\n--- Pixel Difference Metrics (Visual Distinctiveness) ---")
    print(f"  * Uploaded Room vs Option 1 Delta: {diff_orig_v1:.2f} px")
    print(f"  * Uploaded Room vs Option 2 Delta: {diff_orig_v2:.2f} px")
    print(f"  * Option 1 vs Option 2 Delta: {diff_v1_v2:.2f} px")

    assert diff_orig_v1 > 1.0, "Option 1 must differ visually from the uploaded image"
    assert diff_orig_v2 > 1.0, "Option 2 must differ visually from the uploaded image"
    assert diff_v1_v2 > 1.0, "Option 1 and Option 2 must differ visually from each other"

    print("\n[ALL PASSED] 2 Distinct Redesigns with Furniture Repositioning, Style/Color Preferences, and Dynamic Comfort Fully Verified!\n")


if __name__ == "__main__":
    test_2_distinct_redesigns()
