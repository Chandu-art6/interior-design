"""
Test Verification: Multi-User Compromise Consensus & Generation Pipeline
Asserts that changing user styles, priority objects, and colors dynamically updates
the Consensus Engine specifications and inpainting variations across all active inhabitants.
"""
from PIL import Image
import numpy as np
from services.consensus_engine import consensus_engine
from services.generator import generator_service

def test_multi_user():
    print("=" * 70)
    print("=== TESTING MULTI-USER COMPROMISE SYNTHESIS & INPAINTING PIPELINE ===")
    print("=" * 70)

    test_img = np.full((1024, 1024, 3), 200, dtype=np.uint8)
    pil_img = Image.fromarray(test_img)
    mask = Image.fromarray(np.full((1024, 1024), 255, dtype=np.uint8))

    users_test = [
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
        },
        {
            "name": "Taylor",
            "preferred_style": "Scandinavian",
            "priority_objects": ["Bouclé Armchair", "Pendant Lamp"],
            "color_tone": "#38bdf8"
        }
    ]

    # 1. Test Consensus Engine with 3 Users
    consensus_res = consensus_engine.synthesize(
        users=users_test,
        room_type="Living Room",
        budget_tier="Mid-Range Premium",
        budget_limit=24000.0
    )

    summary = consensus_res["consensus_summary"]
    catalog = consensus_res["allocated_catalog"]
    specs = consensus_res["variation_specs"]

    print(f"Consensus Inhabitant Count: {summary['num_inhabitants']}")
    print(f"Synthesized Style Title: {summary['synthesized_style']}")
    print(f"Total Priority Pieces Allocated: {summary['total_priority_pieces']}")
    for cat_item in catalog:
        print(f"  * [{cat_item['requested_by']}] {cat_item['name']}: ${cat_item['estimated_cost']:,.2f} ({cat_item['material_spec'][:40]}...)")

    assert summary["num_inhabitants"] == 3, "Must track all 3 inhabitants"
    assert len(specs) == 4, "Must generate 4 variation specifications"
    assert "Alex" in specs[0]["title"] and "Industrial" in specs[0]["title"], "Var 1 must focus on Alex (Industrial)"
    assert "Sam" in specs[1]["title"] and "Japandi" in specs[1]["title"], "Var 2 must focus on Sam (Japandi)"
    assert "50/50 Hybrid" in specs[2]["title"], "Var 3 must be a 50/50 Hybrid"
    assert "Taylor" in specs[3]["title"] or "Full Group" in specs[3]["title"], "Var 4 must be Full Group Consensus"

    # 2. Test Inpainting Generation with 3 Users
    inpaint_res = generator_service.inpaint(
        image=pil_img,
        mask=mask,
        users=users_test,
        budget_tier="Mid-Range Premium",
        budget_limit=24000.0,
        num_variations=4
    )

    variations = inpaint_res["variations"]
    assert len(variations) == 4, "Must generate 4 variations"

    print("\n--- Generated Variations Output ---")
    for v in variations:
        print(f"\n[{v['variation_id']}] {v['title']}")
        print(f"  * Persona Blend: {v['persona_blend']}")
        print(f"  * Suggested Objects: {v['suggested_objects']}")
        print(f"  * Placements Computed: {len(v.get('placement_suggestions', []))} items")
        assert "image" in v and v["image"].size == (1024, 1024), "Valid image required"
        assert "placement_overlay_image" in v, "Valid placement overlay required"
        assert len(v["placement_suggestions"]) > 0, "Placement suggestions must be computed"

    print("\n" + "=" * 70)
    print("[ALL PASSED] Multi-User consensus synthesis and inpainting pipeline verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_multi_user()
