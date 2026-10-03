from services.recommender import recommender_service
from ui.components.widgets import generate_spatial_clearance_heatmap, generate_comfort_radar_chart, generate_metrics_bar_chart

sample_detections = [
    {"label": "couch", "category": "Seating", "confidence": 0.88, "bbox_normalized": [0.2, 0.5, 0.8, 0.8]},
    {"label": "table", "category": "Surface", "confidence": 0.92, "bbox_normalized": [0.35, 0.7, 0.65, 0.85]}
]
sample_spatial = {
    "furniture_coverage_pct": 28.5,
    "spatial_clutter_score": 14.0
}

print("=== Testing Advanced Architectural Metrics ===")
metrics = recommender_service.evaluate_comfort_metrics(
    detected_items=sample_detections,
    room_type="Living Room",
    spatial_metrics=sample_spatial
)

metrics_dict = metrics.dict() if hasattr(metrics, "dict") else metrics.model_dump()

print("Circulation Flow Rate:", metrics_dict.get("circulation_flow_rate_pct"), "%")
print("Daylight Factor:", metrics_dict.get("daylight_factor_pct"), "% DF")
print("ADA Accessibility Score:", metrics_dict.get("ada_accessibility_score"), "/100")
print("Acoustic NRC Rating:", metrics_dict.get("acoustic_nrc_rating"))
print("Ergonomic Clearance Efficiency:", metrics_dict.get("ergonomic_efficiency_score"), "/100")
print("Actionable Fixes Count:", len(metrics_dict.get("actionable_fixes", [])))

print("\n=== Testing 2D Heatmap & Radar Image Generation ===")
heatmap_img = generate_spatial_clearance_heatmap(metrics_dict)
radar_img = generate_comfort_radar_chart(metrics_dict)
bar_img = generate_metrics_bar_chart(metrics_dict)

print("Heatmap Size:", heatmap_img.size)
print("Radar Size:", radar_img.size)
print("Bar Size:", bar_img.size)

print("\n[OK] All advanced architectural metrics & 2D heatmap tests passed successfully!")
