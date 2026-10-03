from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from services.recommender import ComfortMetricScore, ArchitecturalFixItem


class DominantColorItem(BaseModel):
    """Extracted color palette item."""
    hex: str
    rgb: List[int]
    percentage: float


class SpatialLayoutMetrics(BaseModel):
    """Spatial coverage, clutter, and distribution metrics."""
    furniture_count: int = 0
    furniture_coverage_pct: float = 0.0
    furniture_density: float = 0.0
    spatial_clutter_score: float = 0.0
    spatial_balance: float = 0.0
    layout_distribution: str = "Spacious"


class FurnitureDetectionItem(BaseModel):
    """Detected furniture item schema."""
    label: str
    category: str = "General"
    confidence: float
    bbox_normalized: List[float] = Field(..., description="[x_min, y_min, x_max, y_max] normalized coordinates")
    bbox_pixels: Optional[List[int]] = None
    area_ratio: Optional[float] = None


class ItemChangeItem(BaseModel):
    """Items identified to remove or keep in the room."""
    name: str
    reason: str
    category: Optional[str] = "Decor"
    bbox: Optional[List[float]] = None


class PurchaseSuggestionItem(BaseModel):
    """Structured decor and furniture product purchase suggestion."""
    name: str
    category: str
    estimated_cost: str
    material: str
    placement_rationale: str
    style_match: Optional[str] = None


class PlacementSuggestionItem(BaseModel):
    """Spatial placement coordinates and clearance metrics for recommended pieces."""
    item_name: str
    placement_zone: str
    bbox_normalized: List[float]
    bbox_pixels: Optional[List[int]] = None
    dimensions_cm: Optional[Dict[str, int]] = None
    clearance_corridor_cm: Optional[float] = None
    ada_clearance_compliant: Optional[bool] = True
    orientation: Optional[str] = None


class InpaintAnalysisResult(BaseModel):
    """Structured output from the Vision LLM Analyzer."""
    items_to_remove: List[ItemChangeItem] = []
    items_to_keep: List[ItemChangeItem] = []
    purchase_suggestions: List[PurchaseSuggestionItem] = []
    inpainting_prompt: str
    style_rationale: Optional[str] = None


class RoomAnalysisResponse(BaseModel):
    """Room structural & object analysis schema."""
    analysis_uid: str
    image_url: str
    room_type: str
    detected_objects: List[FurnitureDetectionItem] = []
    spatial_metrics: SpatialLayoutMetrics
    dominant_palette: List[DominantColorItem] = []
    annotated_image_url: Optional[str] = None
    lighting_condition: str = "Normal"


class UserProfilePreference(BaseModel):
    """Individual co-inhabitant design preferences."""
    name: str = "Person 1"
    preferred_style: str = "Modern Minimalist"
    priority_objects: List[str] = []
    color_tone: Optional[str] = "#3b82f6"


class DesignVariationItem(BaseModel):
    """Alternative rendered interior design variation with persona blend info."""
    variation_id: str
    title: str
    subtitle: Optional[str] = ""
    style: str
    persona_blend: Optional[str] = ""
    image_url: str
    placement_overlay_url: Optional[str] = None
    floorplan_diagram_url: Optional[str] = None
    layout_archetype: Optional[str] = None
    wall_anchor: Optional[str] = None
    circulation_corridor: Optional[str] = None
    layout_description: str
    suggested_objects: List[str] = []
    placement_suggestions: Optional[List[Dict[str, Any]]] = []
    estimated_cost: Optional[float] = None
    seed: Optional[int] = None
    controlnet_weight: Optional[float] = None
    pipeline_mode: str = "inpaint"


class BudgetItemAllocation(BaseModel):
    """Allocation breakdown for furniture, lighting, and renovation."""
    furniture_seating_pct: float = 50.0
    lighting_decor_pct: float = 25.0
    renovation_finishes_pct: float = 25.0


class DesignRedesignRequest(BaseModel):
    """Request schema for targeted room redesign and inpainting."""
    image_filename: str
    room_type: str = "Living Room"
    interior_style: str = "Modern Minimalist"
    budget_tier: str = "Mid-Range Premium"
    budget_limit: Optional[float] = 15000.0
    budget_allocation: Optional[BudgetItemAllocation] = None
    users: List[UserProfilePreference] = []
    color_palette: Optional[List[str]] = []
    preserve_structure: bool = True
    custom_prompt_notes: Optional[str] = ""
    target_items_to_replace: Optional[List[str]] = None
    controlnet_scale: float = 0.45
    inference_steps: int = 30
    num_variations: int = 2


class DesignResponse(BaseModel):
    """Complete response schema returning redesigned image, masks, recommendations, and variations."""
    project_uid: str
    original_image_url: str
    redesigned_image_url: str
    primary_generated_url: str  # Alias for backward compatibility
    placement_overlay_url: Optional[str] = None
    floorplan_diagram_url: Optional[str] = None
    mask_image_url: Optional[str] = None
    depth_image_url: Optional[str] = None
    canny_condition_url: Optional[str] = None
    depth_condition_url: Optional[str] = None
    style_applied: str
    room_type: str
    budget_tier: str
    removed_items: List[ItemChangeItem] = []
    kept_items: List[ItemChangeItem] = []
    recommendations: List[PurchaseSuggestionItem] = []
    placement_suggestions: Optional[List[Dict[str, Any]]] = []
    inpainting_prompt: Optional[str] = None
    users: List[UserProfilePreference] = []
    prompt_used: Dict[str, str] = {}
    comfort_metrics: ComfortMetricScore
    explainable_recommendations: List[str] = []
    detected_furniture: List[FurnitureDetectionItem] = []
    variations: List[DesignVariationItem] = []
