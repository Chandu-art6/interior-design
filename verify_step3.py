"""
Comprehensive Verification Script for Step 3 Components
Tests:
1. YOLOv8 Room Analysis & Spatial Layout Metrics
2. ControlNet Conditioning, Prompt Synthesis, and Multi-Variation Generation
3. Recommender Comfort Metrics & Explainable Rationales
4. FastAPI Endpoints End-to-End Test (Upload -> Analyze -> Redesign -> History)
"""
import io
import sys
import numpy as np
from PIL import Image
from pathlib import Path

from config import settings


def test_yolo_service():
    print("\n--- 1. Testing YOLO Room Analysis & Furniture Detection ---")
    from services.yolo_service import yolo_service
    
    # Create sample room image (1024x1024) with colored furniture patches
    test_img = np.full((1024, 1024, 3), 240, dtype=np.uint8)  # Off-white room
    # Draw floor
    test_img[600:, :] = [210, 180, 140]  # Wood floor
    # Draw couch
    test_img[550:850, 150:850] = [60, 70, 90]  # Navy couch
    # Draw coffee table
    test_img[750:900, 350:650] = [139, 69, 19]  # Oak coffee table
    
    pil_img = Image.fromarray(test_img)
    
    analysis_res = yolo_service.detect_furniture(pil_img)
    detections = analysis_res["detections"]
    metrics = analysis_res["spatial_metrics"]
    
    print(f"[OK] Detected items count: {len(detections)}")
    print(f"[OK] Spatial Coverage: {metrics['furniture_coverage_pct']}%")
    print(f"[OK] Spatial Clutter Score: {metrics['spatial_clutter_score']}/100")
    print(f"[OK] Layout Classification: {metrics['layout_distribution']}")
    
    # Test drawing annotations
    annotated = yolo_service.draw_detections(pil_img, detections)
    assert annotated.size == (1024, 1024), "Annotated image size mismatch"
    print(f"[OK] Generated visual annotation overlay ({annotated.size})")


def test_controlnet_service():
    print("\n--- 2. Testing ControlNet Service & Generation Engine ---")
    from services.controlnet_service import controlnet_service
    
    test_img = Image.new("RGB", (1024, 1024), color=(220, 215, 205))
    
    # Test Canny & Depth conditioning extraction
    canny_map = controlnet_service.extract_canny_conditioning(test_img)
    depth_map = controlnet_service.extract_depth_conditioning(test_img)
    
    assert canny_map.size == (1024, 1024), "Canny map size mismatch"
    assert depth_map.size == (1024, 1024), "Depth map size mismatch"
    print(f"[OK] Structural Canny conditioning map extracted ({canny_map.size})")
    print(f"[OK] Structural Depth conditioning map extracted ({depth_map.size})")
    
    # Test prompt synthesis
    prompts = controlnet_service.generate_prompt(
        room_type="Living Room",
        style="Scandinavian",
        color_palette=["#F5F5DC", "#2F4F4F", "#D2B48C"],
        custom_notes="large floor plants, light oak floor"
    )
    assert "scandinavian" in prompts["positive_prompt"].lower(), "Prompt synthesis missing style"
    print(f"[OK] Synthesized Diffusion Positive Prompt: {prompts['positive_prompt'][:90]}...")
    print(f"[OK] Synthesized Diffusion Negative Prompt: {prompts['negative_prompt'][:70]}...")
    
    # Test multi-variation generation
    gen_result = controlnet_service.generate_redesign_variations(
        original_image=test_img,
        room_type="Living Room",
        style="Modern Minimalist",
        num_variations=3
    )
    
    assert gen_result["primary_image"].size == (1024, 1024)
    assert len(gen_result["variations"]) == 3, f"Expected 3 variations, got {len(gen_result['variations'])}"
    print(f"[OK] Primary redesign rendered: {gen_result['primary_image'].size}")
    for var in gen_result["variations"]:
        print(f"  -> Variation '{var['title']}' ({var['variation_id']}): {var['image'].size}")


def test_recommender_service():
    print("\n--- 3. Testing Explainable Recommendations & Comfort Metrics ---")
    from services.recommender import recommender_service
    
    sample_detections = [
        {"label": "couch", "category": "Seating", "confidence": 0.94, "bbox_normalized": [0.1, 0.4, 0.9, 0.8]},
        {"label": "table", "category": "Surface", "confidence": 0.88, "bbox_normalized": [0.3, 0.6, 0.7, 0.85]}
    ]
    spatial_metrics = {
        "furniture_coverage_pct": 28.5,
        "spatial_clutter_score": 18.0
    }
    
    scores = recommender_service.evaluate_comfort_metrics(sample_detections, "Living Room", spatial_metrics)
    print(f"[OK] Circulation Score: {scores.circulation_score}/100")
    print(f"[OK] Lighting Score: {scores.lighting_score}/100")
    print(f"[OK] Ergonomics Score: {scores.ergonomics_score}/100")
    print(f"[OK] Clutter Index: {scores.clutter_index}%")
    print(f"[OK] Overall Comfort Score: {scores.overall_comfort_score}/100")
    
    recs = recommender_service.generate_recommendations(scores, "Living Room", "Japandi", sample_detections)
    print(f"[OK] Generated {len(recs)} explainable architectural justifications:")
    for r in recs:
        print(f"  * {r}")


def test_fastapi_endpoints():
    print("\n--- 4. Testing FastAPI Endpoints End-to-End ---")
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    
    # 1. Health check
    h_resp = client.get("/health")
    assert h_resp.status_code == 200, f"Health check failed: {h_resp.text}"
    print(f"[OK] GET /health returned 200 OK: {h_resp.json()['status']}")
    
    # 2. Upload image
    test_img = Image.new("RGB", (800, 600), color=(200, 190, 180))
    img_byte_arr = io.BytesIO()
    test_img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    upload_resp = client.post(
        "/api/v1/upload",
        files={"file": ("test_room.jpg", img_byte_arr, "image/jpeg")}
    )
    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    uploaded_filename = upload_resp.json()["filename"]
    print(f"[OK] POST /api/v1/upload returned: {uploaded_filename}")
    
    # 3. Analyze Room
    analyze_resp = client.post(
        "/api/v1/analyze",
        data={"image_filename": uploaded_filename, "room_type": "Living Room"}
    )
    assert analyze_resp.status_code == 200, f"Analyze failed: {analyze_resp.text}"
    ana_data = analyze_resp.json()
    print(f"[OK] POST /api/v1/analyze returned UID: {ana_data['analysis_uid']}")
    print(f"  -> Dominant colors: {[c['hex'] for c in ana_data['dominant_palette'][:3]]}")
    print(f"  -> Clutter Score: {ana_data['spatial_metrics']['spatial_clutter_score']}")
    
    # 4. Redesign Space
    redesign_payload = {
        "image_filename": uploaded_filename,
        "room_type": "Living Room",
        "interior_style": "Scandinavian",
        "color_palette": ["#F5F5DC", "#2F4F4F"],
        "preserve_structure": True,
        "custom_prompt_notes": "Warm lighting with oak accents",
        "controlnet_scale": 0.9,
        "inference_steps": 25,
        "num_variations": 3
    }
    redesign_resp = client.post("/api/v1/redesign", json=redesign_payload)
    assert redesign_resp.status_code == 200, f"Redesign failed: {redesign_resp.text}"
    des_data = redesign_resp.json()
    print(f"[OK] POST /api/v1/redesign returned Project UID: {des_data['project_uid']}")
    print(f"  -> Primary image: {des_data['primary_generated_url']}")
    print(f"  -> Overall Comfort Score: {des_data['comfort_metrics']['overall_comfort_score']}")
    print(f"  -> Variations generated: {len(des_data['variations'])}")
    
    # 5. List Projects
    list_resp = client.get("/api/v1/projects")
    assert list_resp.status_code == 200, f"List projects failed: {list_resp.text}"
    print(f"[OK] GET /api/v1/projects returned {list_resp.json()['count']} historical records")


if __name__ == "__main__":
    test_yolo_service()
    test_controlnet_service()
    test_recommender_service()
    test_fastapi_endpoints()
    print("\n=======================================================")
    print("ALL STEP 3 SERVICES & API TESTS COMPLETED SUCCESSFULLY!")
    print("=======================================================")
