"""
Comprehensive Verification Script for Step 2 Components
Tests:
1. Database Schema Initialization (User, Preference, RoomAnalysis, GeneratedDesign)
2. Image Processor (Standardize 1024x1024, Bilateral Filtering, Dominant Color Palette Extraction)
3. Hand Gesture Recognition (Landmark geometry classification & UI trigger mapping)
"""
import sys
import numpy as np
from PIL import Image

def test_image_processor():
    print("\n--- Testing Image Processor ---")
    from services.image_processor import image_processor
    
    # Generate synthetic test image (Color patches with simulated textures)
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    test_img[:300, :400] = [210, 180, 140]  # Tan / Wood
    test_img[300:, :400] = [47, 79, 79]    # Dark Slate
    test_img[:, 400:] = [245, 245, 220]    # Beige
    
    result = image_processor.preprocess_pipeline(test_img, target_size=(1024, 1024), palette_clusters=3)
    
    std_img = result["standardized_image"]
    filt_img = result["filtered_image"]
    palette = result["dominant_palette"]
    
    assert std_img.size == (1024, 1024), f"Expected (1024, 1024), got {std_img.size}"
    assert filt_img.size == (1024, 1024), f"Expected (1024, 1024), got {filt_img.size}"
    assert len(palette) > 0, "Palette extraction returned empty list"
    
    print(f"[OK] Image standardized to: {std_img.size}")
    print(f"[OK] Bilateral filtered image generated: {filt_img.size}")
    print(f"[OK] Top extracted colors: {[p['hex'] + ' (' + str(p['percentage']) + '%)' for p in palette]}")


def test_gesture_detector():
    print("\n--- Testing Gesture Detector & Gesture Classifications ---")
    from services.gesture_detector import gesture_detector, LandmarkPoint
    
    # 1. Test Open Palm (All 4 fingers extended)
    # y=0 is top, y=1 is bottom; extended finger has tip.y < pip.y < mcp.y
    open_palm_landmarks = [
        LandmarkPoint(0.5, 0.9, 0.0),    # 0: Wrist
        LandmarkPoint(0.4, 0.8, 0.0),    # 1: Thumb CMC
        LandmarkPoint(0.35, 0.7, 0.0),   # 2: Thumb MCP
        LandmarkPoint(0.3, 0.6, 0.0),    # 3: Thumb IP
        LandmarkPoint(0.25, 0.5, 0.0),   # 4: Thumb TIP
        LandmarkPoint(0.45, 0.6, 0.0),   # 5: Index MCP
        LandmarkPoint(0.45, 0.4, 0.0),   # 6: Index PIP
        LandmarkPoint(0.45, 0.3, 0.0),   # 7: Index DIP
        LandmarkPoint(0.45, 0.2, 0.0),   # 8: Index TIP (Extended)
        LandmarkPoint(0.5, 0.6, 0.0),    # 9: Middle MCP
        LandmarkPoint(0.5, 0.4, 0.0),    # 10: Middle PIP
        LandmarkPoint(0.5, 0.3, 0.0),    # 11: Middle DIP
        LandmarkPoint(0.5, 0.18, 0.0),   # 12: Middle TIP (Extended)
        LandmarkPoint(0.55, 0.6, 0.0),   # 13: Ring MCP
        LandmarkPoint(0.55, 0.4, 0.0),   # 14: Ring PIP
        LandmarkPoint(0.55, 0.3, 0.0),   # 15: Ring DIP
        LandmarkPoint(0.55, 0.22, 0.0),  # 16: Ring TIP (Extended)
        LandmarkPoint(0.6, 0.65, 0.0),   # 17: Pinky MCP
        LandmarkPoint(0.6, 0.5, 0.0),    # 18: Pinky PIP
        LandmarkPoint(0.6, 0.4, 0.0),    # 19: Pinky DIP
        LandmarkPoint(0.6, 0.3, 0.0),    # 20: Pinky TIP (Extended)
    ]
    
    g_name, g_action, conf = gesture_detector.classify_gesture(open_palm_landmarks)
    assert g_name == "OPEN_PALM" and g_action == "RESET_VIEW", f"Expected OPEN_PALM, got {g_name} / {g_action}"
    print(f"[OK] Classified OPEN_PALM -> Action: '{g_action}' (Confidence: {conf})")

    # 2. Test Pinch Gesture (Thumb tip close to Index tip)
    pinch_landmarks = list(open_palm_landmarks)
    pinch_landmarks[4] = LandmarkPoint(0.44, 0.21, 0.0)  # Thumb tip right next to index tip
    pinch_landmarks[8] = LandmarkPoint(0.45, 0.20, 0.0)  # Index tip
    g_name, g_action, conf = gesture_detector.classify_gesture(pinch_landmarks)
    assert g_name == "PINCH" and g_action == "SELECT_OBJECT", f"Expected PINCH, got {g_name} / {g_action}"
    print(f"[OK] Classified PINCH -> Action: '{g_action}' (Confidence: {conf})")

    # 3. Test Thumbs Up (Thumb up, all 4 fingers curled)
    thumbs_up_landmarks = list(open_palm_landmarks)
    # Curl index, middle, ring, pinky (tip.y > pip.y)
    thumbs_up_landmarks[4] = LandmarkPoint(0.3, 0.2, 0.0)   # Thumb TIP pointing UP
    thumbs_up_landmarks[2] = LandmarkPoint(0.3, 0.7, 0.0)   # Thumb MCP lower
    thumbs_up_landmarks[8] = LandmarkPoint(0.45, 0.65, 0.0)  # Index TIP curled
    thumbs_up_landmarks[6] = LandmarkPoint(0.45, 0.55, 0.0)  # Index PIP
    thumbs_up_landmarks[12] = LandmarkPoint(0.5, 0.65, 0.0)  # Middle TIP curled
    thumbs_up_landmarks[10] = LandmarkPoint(0.5, 0.55, 0.0)  # Middle PIP
    thumbs_up_landmarks[16] = LandmarkPoint(0.55, 0.65, 0.0) # Ring TIP curled
    thumbs_up_landmarks[14] = LandmarkPoint(0.55, 0.55, 0.0) # Ring PIP
    thumbs_up_landmarks[20] = LandmarkPoint(0.6, 0.65, 0.0)  # Pinky TIP curled
    thumbs_up_landmarks[18] = LandmarkPoint(0.6, 0.55, 0.0)  # Pinky PIP
    g_name, g_action, conf = gesture_detector.classify_gesture(thumbs_up_landmarks)
    assert g_name == "THUMBS_UP" and g_action == "APPROVE_DESIGN", f"Expected THUMBS_UP, got {g_name} / {g_action}"
    print(f"[OK] Classified THUMBS_UP -> Action: '{g_action}' (Confidence: {conf})")


def test_database():
    print("\n--- Testing Database Layer ---")
    from database.session import init_db, engine, SessionLocal
    from database.models import User, Preference, RoomAnalysis, GeneratedDesign
    from sqlalchemy import inspect

    init_db()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"[OK] Database tables verified: {tables}")
    
    required_tables = {"users", "preferences", "room_analyses", "generated_designs"}
    for t in required_tables:
        assert t in tables, f"Missing expected table: {t}"

    db = SessionLocal()
    try:
        # Create test user
        user = User(username="architect_step2", email="architect_step2@studio.ai")
        db.add(user)
        db.commit()
        db.refresh(user)

        # Create user preference
        pref = Preference(
            user_id=user.id,
            budget_tier="luxury",
            style_preferences=["Scandinavian", "Japandi"],
            color_preferences=["#D2B48C", "#F5F5DC"],
            multi_user_weighting={"user_1": 0.7, "user_2": 0.3}
        )
        db.add(pref)

        # Create room analysis
        analysis = RoomAnalysis(
            analysis_uid="ana-step2-001",
            user_id=user.id,
            image_path="/static/uploads/living_room.jpg",
            room_type="Living Room",
            detected_objects=[
                {"label": "couch", "confidence": 0.94, "bbox": [0.15, 0.3, 0.85, 0.75]},
                {"label": "coffee table", "confidence": 0.88, "bbox": [0.35, 0.6, 0.65, 0.85]}
            ],
            spatial_dimensions={"width_m": 5.0, "length_m": 6.2, "area_sqm": 31.0},
            lighting_condition="High Natural Light",
            clutter_score=15.5
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        # Create generated design
        design = GeneratedDesign(
            design_uid="des-step2-001",
            user_id=user.id,
            room_analysis_id=analysis.id,
            style_applied="Scandinavian",
            original_image_path="/static/uploads/living_room.jpg",
            generated_image_path="/static/outputs/living_room_scandinavian.jpg",
            prompt_used="Scandinavian minimalist living room with warm light wood, linen sofa, natural daylight",
            circulation_score=92.0,
            lighting_score=95.0,
            ergonomics_score=90.0,
            clutter_index=12.0,
            overall_comfort_score=93.5,
            ai_reasoning_log=[
                "Maintained unobstructed 95cm pathway from doorway to seating.",
                "Maximized southern window natural illumination.",
                "Applied warm neutral color palette with high reflectance value."
            ]
        )
        db.add(design)
        db.commit()
        db.refresh(design)

        print(f"[OK] User created: ID={user.id}, Username='{user.username}'")
        print(f"[OK] Preference created: Tier='{pref.budget_tier}', Multi-user weighting={pref.multi_user_weighting}")
        print(f"[OK] RoomAnalysis created: UID='{analysis.analysis_uid}', Detections={len(analysis.detected_objects)}")
        print(f"[OK] GeneratedDesign created: UID='{design.design_uid}', Comfort Score={design.overall_comfort_score}")

    finally:
        db.close()


if __name__ == "__main__":
    test_image_processor()
    test_gesture_detector()
    test_database()
    print("\n==========================================")
    print("ALL STEP 2 VERIFICATIONS PASSED 100%!")
    print("==========================================")
