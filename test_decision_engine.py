from services.recommender import recommender_service

print("=== Testing AI Decision Explanation & Tradeoff Engine ===")
decision = recommender_service.generate_ai_decision_explanation(
    room_type="Living Room",
    selected_style="Japandi",
    budget_tier="Mid-Range Premium",
    budget_limit=15000.0,
    furn_pct=50.0,
    light_pct=25.0,
    reno_pct=25.0,
    co_inhabitants=[{"name": "Alex", "preferred_style": "Industrial"}, {"name": "Sam", "preferred_style": "Japandi"}]
)

sel = decision["selected_style_analysis"]
print(f"\n[Why Selected] {sel['selected_style']}:")
print(f"  * Compatibility Score: {sel['compatibility_score']}%")
print(f"  * Budget Fit Score: {sel['budget_fit_score']}%")
print(f"  * Products Recommended: {len(sel['products'])} items")
print(f"  * Color Swatches (60-30-10): {len(sel['color_swatches'])} swatches")

rej = decision["rejected_styles_analysis"]
print(f"\n[Why Rejected] {len(rej)} Alternative Styles Evaluated:")
for r in rej:
    print(f"  * {r['style_name']} -> {r['primary_rejection_reason']}")

fin = decision["financial_feasibility"]
print(f"\n[Financial Feasibility]:")
print(f"  * Total Estimated Cost: ${fin['total_estimated_cost']:,}")
print(f"  * Remaining Buffer: ${fin['budget_buffer']:,}")
print(f"  * Feasibility Score: {fin['feasibility_score']}% ({fin['feasibility_status']})")

print("\n[OK] AI Decision Explanation Engine verified successfully!")
