"""
Test Variation Diversity and Pixel Preservation.
Verifies that all 4 variations are visually distinct from one another,
while locking the unmasked room architecture.
"""
from PIL import Image
import numpy as np
from pathlib import Path

from config import settings
from services.generator import generator_service
from services.yolo_service import yolo_service
from services.segmenter import segmenter_service

def test_variation_diversity():
    print("=" * 70)
    print("=== TESTING VARIATION VISUAL DIVERSITY & PIXEL PRESERVATION ===")
    print("=" * 70)

    # 1. Load user uploaded room
    artifact_dir = Path(r"C:\Users\G chandu\.gemini\antigravity-ide\brain\7698e2b7-70a0-419d-82d8-672ab869cc86")
    img_path = artifact_dir / ".user_uploaded" / "media_1787635096809.png"
    if not img_path.exists():
        img_path = settings.UPLOAD_DIR / "sample_room.jpg"
    
    orig_img = Image.open(str(img_path)).convert("RGB")
    w, h = orig_img.size
    orig_np = np.array(orig_img)

    # 2. Mask: top 30% of room is unmasked architecture (ceiling, upper walls, windows)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[int(h * 0.35):, :] = 255
    mask = Image.fromarray(mask_np)

    users = [
        {"name": "Alex", "preferred_style": "Industrial", "priority_objects": ["Leather Sofa", "Steel Bookshelf"], "color_tone": "#475569"},
        {"name": "Sam", "preferred_style": "Japandi", "priority_objects": ["Monstera Plant", "Floor Lamp"], "color_tone": "#e29578"}
    ]

    res = generator_service.inpaint(
        image=orig_img,
        mask=mask,
        style="Industrial",
        users=users,
        budget_tier="Mid-Range Premium",
        budget_limit=20000.0,
        num_variations=4
    )

    variations = res["variations"]
    assert len(variations) == 4, "Must generate 4 variations"

    var_images = [np.array(v["image"]) for v in variations]

    # Verify that all pairs of variations are visually distinct
    print("\n--- Cross-Variation Visual Diversity Matrix ---")
    for i in range(4):
        for j in range(i + 1, 4):
            diff = np.abs(var_images[i].astype(int) - var_images[j].astype(int))
            mean_diff = diff.mean()
            max_diff = diff.max()
            print(f"  * Variation {i+1} vs Variation {j+1}: Mean pixel delta = {mean_diff:.2f}, Max delta = {max_diff}")
            assert mean_diff > 1.0, f"Variation {i+1} and {j+1} MUST NOT be pixel-identical!"

    # Verify that unmasked architecture (top 30% of image where mask is 0) is 100% bitwise preserved
    unmasked = mask_np == 0
    for idx, v_img in enumerate(var_images):
        arch_diff = np.abs(orig_np[unmasked].astype(int) - v_img[unmasked].astype(int)).max()
        print(f"  * Variation {idx+1} Upper Architecture Preservation: Max delta = {arch_diff} (Target: 0)")
        assert arch_diff == 0, f"Variation {idx+1} MUST 100% preserve unmasked architecture!"

    print("\n" + "=" * 70)
    print("[ALL PASSED] All 4 variations are visually distinct with 100% architecture lock!")
    print("=" * 70)

if __name__ == "__main__":
    test_variation_diversity()
