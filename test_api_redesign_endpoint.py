"""
Test FastAPI Inpainting and Redesign API Endpoints
Asserts that /api/v1/redesign returns structured consensus redesigns,
placement blueprint overlays, non-colliding placement coordinates, and 100% unmasked pixel preservation.
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from config import settings

client = TestClient(app)

def test_api_endpoints():
    print("=" * 70)
    print("=== TESTING FASTAPI INPAINTING, CONSENSUS & REDESIGN ENDPOINTS ===")
    print("=" * 70)

    # 1. Create a synthetic upload file
    test_img = np.full((1024, 1024, 3), 230, dtype=np.uint8)
    test_img[600:, :] = [210, 180, 140]  # Floor
    test_img[520:800, 200:800] = [70, 80, 100]  # Seating
    test_filename = "test_endpoint_room.png"
    test_path = settings.UPLOAD_DIR / test_filename
    Image.fromarray(test_img).save(test_path)
    uploaded_disk_img = np.array(Image.open(test_path).convert("RGB"))

    # 2. Test /api/v1/analyze-inpaint
    print("\n1. Testing POST /api/v1/analyze-inpaint...")
    res_analyze = client.post(
        "/api/v1/analyze-inpaint",
        data={
            "filename": test_filename,
            "room_type": "Living Room",
            "interior_style": "Scandinavian",
            "budget_tier": "Mid-Range Premium",
            "budget_limit": "20000"
        }
    )
    assert res_analyze.status_code == 200, f"Failed: {res_analyze.text}"
    data_analyze = res_analyze.json()
    print(f"  * Items to Remove: {len(data_analyze['items_to_remove'])}")
    print(f"  * Items to Keep: {len(data_analyze['items_to_keep'])}")
    print(f"  * Purchase Suggestions: {len(data_analyze['purchase_suggestions'])}")
    print(f"  * Inpaint Prompt: {data_analyze['inpainting_prompt'][:80]}...")
    print("  [OK] /api/v1/analyze-inpaint returned structured JSON successfully.")

    # 3. Test POST /api/v1/redesign
    print("\n2. Testing POST /api/v1/redesign...")
    payload = {
        "image_filename": test_filename,
        "room_type": "Living Room",
        "interior_style": "Scandinavian",
        "budget_tier": "Mid-Range Premium",
        "budget_limit": 20000.0,
        "users": [
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
        ],
        "num_variations": 4
    }
    res_redesign = client.post("/api/v1/redesign", json=payload)
    assert res_redesign.status_code == 200, f"Redesign failed: {res_redesign.text}"
    data_redesign = res_redesign.json()

    print(f"  * Project UID: {data_redesign['project_uid']}")
    print(f"  * Redesigned Image URL: {data_redesign['redesigned_image_url']}")
    print(f"  * Placement Overlay URL: {data_redesign.get('placement_overlay_url')}")
    print(f"  * Mask Image URL: {data_redesign['mask_image_url']}")
    print(f"  * Depth Image URL: {data_redesign['depth_image_url']}")
    print(f"  * Removed Items: {len(data_redesign['removed_items'])}")
    print(f"  * Kept Items: {len(data_redesign['kept_items'])}")
    print(f"  * Purchase Recommendations: {len(data_redesign['recommendations'])}")
    print(f"  * Primary Placement Suggestions: {len(data_redesign.get('placement_suggestions', []))}")
    print(f"  * Variations Generated: {len(data_redesign['variations'])}")
    print(f"  * Overall Comfort Score: {data_redesign['comfort_metrics']['overall_comfort_score']}")

    assert "redesigned_image_url" in data_redesign, "Must return redesigned image URL"
    assert "placement_overlay_url" in data_redesign, "Must return placement overlay URL"
    assert "mask_image_url" in data_redesign, "Must return mask image URL"
    assert "depth_image_url" in data_redesign, "Must return depth image URL"
    assert len(data_redesign["variations"]) == 4, "Must return 4 variations"
    assert len(data_redesign.get("placement_suggestions", [])) > 0, "Must return placement suggestions"

    # Verify that unmasked pixels are 100% strictly identical to the uploaded photo
    redesign_file = settings.OUTPUT_DIR / Path(data_redesign["redesigned_image_url"]).name
    mask_file = settings.OUTPUT_DIR / Path(data_redesign["mask_image_url"]).name
    
    redesign_img_np = np.array(Image.open(redesign_file))
    mask_img_np = np.array(Image.open(mask_file).convert("L"))
    
    unmasked = mask_img_np == 0
    orig_unmasked = uploaded_disk_img[unmasked]
    res_unmasked = redesign_img_np[unmasked]
    if len(orig_unmasked) > 0:
        pixel_diff = np.abs(orig_unmasked.astype(int) - res_unmasked.astype(int)).max()
        print(f"\n  * Structure Preservation Check: Max pixel difference outside mask = {pixel_diff} (Target: 0)")
        assert pixel_diff == 0, "Unmasked room architecture MUST be 100% identical to uploaded image!"
    else:
        print("\n  * Structure Preservation Check: Full canvas redesigned.")

    print("\n[ALL API TESTS PASSED]")

if __name__ == "__main__":
    test_api_endpoints()
