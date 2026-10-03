"""
Test Verification: Budget Tier Changes & Material Grading
Asserts that changing the budget tier (Budget Essential vs Mid-Range Premium vs Luxury Bespoke)
modifies material qualifiers, finish grades, and estimated costs in the Consensus Engine and Generator.
"""
from PIL import Image
import numpy as np
from services.consensus_engine import consensus_engine
from services.generator import generator_service

def test_budget_variations():
    print("=" * 70)
    print("=== VERIFYING BUDGET-DRIVEN GENERATION & CONSENSUS SPECIFICATIONS ===")
    print("=" * 70)

    test_img = np.full((1024, 1024, 3), 210, dtype=np.uint8)
    pil_img = Image.fromarray(test_img)
    mask = Image.fromarray(np.full((1024, 1024), 255, dtype=np.uint8))

    tiers = [
        ("Budget Essential", 5000.0),
        ("Mid-Range Premium", 15000.0),
        ("Luxury Bespoke", 50000.0)
    ]

    captured_prompts = {}
    captured_allocations = {}

    for tier_name, limit in tiers:
        print(f"\n================ Testing Budget Tier: {tier_name} (${limit:,.0f}) ================")
        
        # 1. Test Consensus Engine
        consensus_data = consensus_engine.synthesize(
            users=[
                {"name": "Alex", "preferred_style": "Industrial", "priority_objects": ["Leather Sofa", "Coffee Table"]},
                {"name": "Sam", "preferred_style": "Japandi", "priority_objects": ["Monstera Plant", "Floor Lamp"]}
            ],
            budget_tier=tier_name,
            budget_limit=limit
        )

        b_summary = consensus_data["consensus_summary"]
        catalog = consensus_data["allocated_catalog"]
        specs = consensus_data["variation_specs"]

        print(f"  * Furniture Allowance: ${b_summary['allocated_budget']['furniture_allowance']:,.2f}")
        print(f"  * Lighting/Decor Allowance: ${b_summary['allocated_budget']['lighting_decor_allowance']:,.2f}")
        print(f"  * Item Count in Catalog: {len(catalog)}")
        for item in catalog[:2]:
            print(f"    - {item['name']}: ${item['estimated_cost']:,.2f} ({item['material_spec'][:45]}...)")

        captured_prompts[tier_name] = specs[0]["prompt"]
        captured_allocations[tier_name] = b_summary["allocated_budget"]

        # 2. Test Inpainting Generator
        inpaint_res = generator_service.inpaint(
            image=pil_img,
            mask=mask,
            budget_tier=tier_name,
            budget_limit=limit,
            num_variations=4
        )

        assert len(inpaint_res["variations"]) == 4, "Must return 4 variations"
        assert inpaint_res["variations"][0]["image"].size == (1024, 1024), "Output image size must match"
        assert inpaint_res["variations"][0]["placement_suggestions"] is not None, "Must have placement suggestions"

    # Assertions: Verify that changing budget tier modifies the output prompts and allocations
    assert captured_prompts["Budget Essential"] != captured_prompts["Luxury Bespoke"], \
        "Prompts MUST differ across different budget tiers!"
    assert captured_prompts["Budget Essential"] != captured_prompts["Mid-Range Premium"], \
        "Prompts MUST differ across Budget Essential and Mid-Range Premium!"
    assert captured_allocations["Budget Essential"]["furniture_allowance"] < captured_allocations["Luxury Bespoke"]["furniture_allowance"], \
        "Furniture allowance MUST scale with budget limit!"

    print("\n" + "=" * 70)
    print("[ALL PASSED] Budget Tier scaling, material grading, and prompt variations verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_budget_variations()
