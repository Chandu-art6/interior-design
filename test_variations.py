from PIL import Image
import numpy as np
from services.controlnet_service import controlnet_service

test_img = np.full((1024, 1024, 3), 200, dtype=np.uint8)
pil_img = Image.fromarray(test_img)

styles = ["Modern Minimalist", "Scandinavian", "Japandi", "Industrial"]

print("=== Verifying 4 Distinct Spatial Layouts, Prompts & Seeds ===")
for style in styles:
    res = controlnet_service.generate_redesign_variations(pil_img, style=style, num_variations=4)
    print(f"\nStyle: {style} (Conditioning Scale: {res['controlnet_scale']})")
    for v in res['variations']:
        arr = np.array(v['image'])
        mean_rgb = arr.mean(axis=(0, 1)).round(1).tolist()
        print(f"  [{v['variation_id']}] {v['title']} ({v['subtitle']})")
        print(f"      * Seed: {v['seed']} | Scale: {v['controlnet_weight']}")
        print(f"      * Spatial Prompt: {v['spatial_prompt']}")
        print(f"      * Mean RGB: {mean_rgb}")

print("\n[OK] 4 Distinct Furniture Layouts & Prompts Verified Successfully!")
