from fastapi.testclient import TestClient
from app.main import app
import numpy as np
from PIL import Image
from config import settings
import time

client = TestClient(app)

# 1. Create a dummy image
sample_img = np.full((1024, 1024, 3), 220, dtype=np.uint8)
sample_pil = Image.fromarray(sample_img)
sample_filename = f"test_full_{int(time.time()*1000)}.jpg"
sample_pil.save(settings.UPLOAD_DIR / sample_filename)

# 2. Test Redesign Endpoint
payload = {
    "image_filename": sample_filename,
    "room_type": "Living Room",
    "interior_style": "Japandi",
    "budget_tier": "Mid-Range Premium",
    "budget_limit": 15000.0,
    "users": [
        {"name": "Alex", "preferred_style": "Industrial", "priority_objects": ["Leather Sofa"], "color_tone": "#475569"},
        {"name": "Sam", "preferred_style": "Japandi", "priority_objects": ["Indoor Plants"], "color_tone": "#e29578"}
    ],
    "color_palette": ["#475569", "#e29578"],
    "preserve_structure": True,
    "custom_prompt_notes": "pet friendly",
    "controlnet_scale": 0.45,
    "inference_steps": 30,
    "num_variations": 4
}

print("Testing POST /api/v1/redesign...")
response = client.post("/api/v1/redesign", json=payload)
print("Status Code:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print("Project UID:", data["project_uid"])
    print("Variations returned:", len(data["variations"]))
    print("Comfort Score:", data["comfort_metrics"]["overall_comfort_score"])
    print("Circulation Flow:", data["comfort_metrics"]["circulation_flow_rate_pct"], "%")
    print("\n[OK] Redesign API validation passed with 100% success!")
else:
    print("Error Response:", response.text)
    raise Exception("API call failed")
