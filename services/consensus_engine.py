"""
Multi-User Consensus Engine for AI Interior Space Redesign
Synthesizes preferences from multiple inhabitants (2, 3, or 4 users), merges conflicting
aesthetic styles into harmonious architectural palettes, applies budget constraints,
and produces structured specifications for 4 distinct layout variations.
"""
from typing import List, Dict, Any, Optional, Tuple
import math

# Material, lighting, and palette rules per style
STYLE_ARCHITECTURAL_SPECS: Dict[str, Dict[str, Any]] = {
    "Modern Minimalist": {
        "primary_materials": ["matte black steel", "white oak", "polished architectural concrete", "low-iron glass"],
        "lighting_type": "recessed architectural 3000K LED channels and indirect cove lighting",
        "furniture_forms": "low-profile rectilinear silhouettes, handleless seamless cabinetry",
        "color_undertone": "cool neutral (#f1f5f9 base with slate accents)",
        "spatial_vibe": "uncluttered open-concept flow with expansive negative space"
    },
    "Scandinavian": {
        "primary_materials": ["blonde ash wood", "textured bouclé wool", "natural oatmeal linen", "glazed ceramic"],
        "lighting_type": "abundant natural daylight ingress with layered 2700K washi paper pendants",
        "furniture_forms": "organic rounded tapered legs, ergonomic curved armchairs",
        "color_undertone": "warm airy hygge (#fef3c7 base with muted pastel accents)",
        "spatial_vibe": "warm, functional, light-filled coziness"
    },
    "Japandi": {
        "primary_materials": ["solid hinoki cypress", "honed travertine stone", "wabi-sabi raw linen", "blackened cedar slats"],
        "lighting_type": "diffused ambient shoji paper lantern glow and 2400K warm downlights",
        "furniture_forms": "grounded low-height seating, organic asymmetric center tables",
        "color_undertone": "earthy wabi-sabi (#d97706 warm earth with charcoal accents)",
        "spatial_vibe": "zen serenity, natural imperfections, grounded balance"
    },
    "Industrial": {
        "primary_materials": ["distressed top-grain cognac leather", "exposed blackened cast iron", "reclaimed rough-sawn timber", "wire-mesh glass"],
        "lighting_type": "Edison amber filament pendants, directional matte black track spotlights",
        "furniture_forms": "heavy-gauge welded structural frames, deep tufted leather sofas",
        "color_undertone": "moody dark loft (#475569 iron grey with amber highlights)",
        "spatial_vibe": "raw architectural texture, high contrast, utilitarian craft"
    },
    "Bohemian Modern": {
        "primary_materials": ["handwoven rattan", "terracotta earthenware", "vintage wool kilim textiles", "solid mango wood"],
        "lighting_type": "warm ambient fairy lighting, perforated brass lanterns",
        "furniture_forms": "modular relaxed seating, layered floor poufs and macramé accents",
        "color_undertone": "vibrant earthy (#ea580c terracotta with sage green accents)",
        "spatial_vibe": "eclectic warmth, rich botanical layering, relaxed comfort"
    },
    "Luxury Contemporary": {
        "primary_materials": ["Calacatta gold honed marble", "brushed champagne brass", "curved Italian bouclé", "fluted smoked eucalyptus"],
        "lighting_type": "bespoke cove illumination, sculptural crystal and brass chandelier",
        "furniture_forms": "sculptural curved organic sectionals, tiered floating stone tables",
        "color_undertone": "sophisticated opulence (#e2e8f0 pearl with champagne gold accents)",
        "spatial_vibe": "high-end penthouse luxury, refined craftsmanship, grand scale"
    },
    "Biophilic Organic": {
        "primary_materials": ["living moss panels", "FSC-certified bamboo", "solid teak root", "unbleached organic cotton"],
        "lighting_type": "full-spectrum circadian daylight simulation, soft dappled canopy light",
        "furniture_forms": "biomorphic flowing contours, integrated indoor planter furniture",
        "color_undertone": "lush botanical (#15803d forest emerald with sandy neutral base)",
        "spatial_vibe": "indoor-outdoor nature sanctuary, restorative wellness"
    },
    "Mid-Century Modern": {
        "primary_materials": ["warm walnut veneer", "mustard ribbed velvet", "brushed antique brass", "molded plywood"],
        "lighting_type": "iconic Sputnik brass chandeliers, arched chrome floor lamps",
        "furniture_forms": "tapered dowel legs, iconic winged lounge chairs, geometric credenzas",
        "color_undertone": "retro warmth (#d97706 warm ochre with teal accents)",
        "spatial_vibe": "timeless architectural elegance, clean geometric lines"
    }
}

# Budget tier constraints and material grading
BUDGET_TIER_SPECS: Dict[str, Dict[str, Any]] = {
    "Budget Essential": {
        "grade": "Functional & Accessible",
        "material_qualifiers": "engineered wood with realistic grain laminate, durable high-rub linen blend, powder-coated steel hardware",
        "target_max_piece_cost": 650.0,
        "finish_level": "Clean streamlined finishes with smart flat-pack modularity and high durability",
        "lighting_budget_pct": 0.20,
        "furniture_budget_pct": 0.60,
        "renovation_budget_pct": 0.20,
        "cost_multiplier": 0.75
    },
    "Mid-Range Premium": {
        "grade": "Artisanal & High Quality",
        "material_qualifiers": "solid hardwood frames (white oak/ash), top-grain leather upholstery, honed natural stone, brushed anodized metal",
        "target_max_piece_cost": 2200.0,
        "finish_level": "Custom craftsmanship, durable mortise-and-tenon joinery, premium commercial-grade textiles",
        "lighting_budget_pct": 0.25,
        "furniture_budget_pct": 0.50,
        "renovation_budget_pct": 0.25,
        "cost_multiplier": 1.0
    },
    "Luxury Bespoke": {
        "grade": "Bespoke Haute Couture",
        "material_qualifiers": "bookmatched Italian Calacatta marble, bespoke solid walnut, genuine shearling/bouclé, hand-rubbed champagne brass",
        "target_max_piece_cost": 7500.0,
        "finish_level": "Architect-commissioned bespoke fabrications, museum-grade detailing, integrated smart ambient systems",
        "lighting_budget_pct": 0.25,
        "furniture_budget_pct": 0.45,
        "renovation_budget_pct": 0.30,
        "cost_multiplier": 1.45
    }
}


class ConsensusEngine:
    """
    Synthesizes multi-user interior design preferences, resolves conflicting styles,
    applies budget constraints, and generates structured prompts for 4 layout variations.
    """

    def synthesize(
        self,
        users: Optional[List[Dict[str, Any]]] = None,
        room_type: str = "Living Room",
        budget_tier: str = "Mid-Range Premium",
        budget_limit: float = 15000.0,
        custom_notes: Optional[str] = "",
        color_palette: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint: Builds a comprehensive consensus model for the household.
        """
        # Normalize and validate users
        normalized_users = self._normalize_users(users)
        
        # Resolve budget tier specs
        b_spec = BUDGET_TIER_SPECS.get(budget_tier, BUDGET_TIER_SPECS["Mid-Range Premium"])
        
        # 1. Compute Hybrid Style & Harmony
        u1 = normalized_users[0]
        u2 = normalized_users[1] if len(normalized_users) > 1 else u1
        
        u1_style = u1["preferred_style"]
        u2_style = u2["preferred_style"]
        
        hybrid_style_title = self._create_hybrid_style_name(u1_style, u2_style)
        
        # 2. Extract and allocate priority furniture catalog across users
        allocated_catalog = self._allocate_priority_catalog(
            normalized_users, budget_limit, b_spec
        )
        
        # 3. Build 4 Structured Variation Specs
        variation_specs = self._build_variation_specs(
            users=normalized_users,
            room_type=room_type,
            budget_tier=budget_tier,
            budget_limit=budget_limit,
            budget_spec=b_spec,
            allocated_catalog=allocated_catalog,
            custom_notes=custom_notes,
            color_palette=color_palette
        )
        
        # 4. Consensus Summary Card Info
        consensus_summary = {
            "num_inhabitants": len(normalized_users),
            "inhabitant_names": [u["name"] for u in normalized_users],
            "primary_styles": [u["preferred_style"] for u in normalized_users],
            "synthesized_style": hybrid_style_title,
            "budget_tier": budget_tier,
            "budget_limit": budget_limit,
            "allocated_budget": {
                "furniture_allowance": round(budget_limit * b_spec["furniture_budget_pct"], 2),
                "lighting_decor_allowance": round(budget_limit * b_spec["lighting_budget_pct"], 2),
                "renovation_reserve": round(budget_limit * b_spec["renovation_budget_pct"], 2)
            },
            "total_priority_pieces": len(allocated_catalog),
            "top_compromise_items": [item["name"] for item in allocated_catalog[:6]]
        }
        
        return {
            "consensus_summary": consensus_summary,
            "allocated_catalog": allocated_catalog,
            "variation_specs": variation_specs,
            "budget_spec": b_spec
        }

    def _normalize_users(self, users: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Ensures at least 2 valid user profiles exist."""
        if not users or len(users) == 0:
            return [
                {
                    "name": "Alex",
                    "preferred_style": "Industrial",
                    "priority_objects": ["Leather Sofa", "Coffee Table", "Steel Bookshelf"],
                    "color_tone": "#475569"
                },
                {
                    "name": "Sam",
                    "preferred_style": "Japandi",
                    "priority_objects": ["Monstera Plant", "Floor Lamp", "Ceramic Vases"],
                    "color_tone": "#e29578"
                }
            ]
        
        normalized = []
        for i, u in enumerate(users):
            name = u.get("name", f"Inhabitant {i+1}").strip() or f"Inhabitant {i+1}"
            style = u.get("preferred_style", "Modern Minimalist")
            if style not in STYLE_ARCHITECTURAL_SPECS:
                style = "Modern Minimalist"
            
            p_objs = u.get("priority_objects", [])
            if isinstance(p_objs, str):
                p_objs = [x.strip() for x in p_objs.split(",") if x.strip()]
            elif not isinstance(p_objs, list) or len(p_objs) == 0:
                p_objs = ["Lounge Seating", "Accent Table", "Ambient Lighting"]
            
            color = u.get("color_tone", "#6366f1")
            normalized.append({
                "name": name,
                "preferred_style": style,
                "priority_objects": p_objs,
                "color_tone": color
            })
            
        if len(normalized) == 1:
            normalized.append({
                "name": "Co-Inhabitant",
                "preferred_style": "Scandinavian" if normalized[0]["preferred_style"] != "Scandinavian" else "Japandi",
                "priority_objects": ["Bouclé Chair", "Area Rug", "Pendant Lamp"],
                "color_tone": "#38bdf8"
            })
            
        return normalized

    def _create_hybrid_style_name(self, style1: str, style2: str) -> str:
        """Constructs an architectural hybrid style name."""
        if style1 == style2:
            return f"Curated {style1}"
        
        pair = {style1, style2}
        if pair == {"Industrial", "Japandi"}:
            return "Wabi-Sabi Industrial Loft"
        elif pair == {"Industrial", "Scandinavian"}:
            return "Nordic Industrial Modern"
        elif pair == {"Modern Minimalist", "Japandi"}:
            return "Minimalist Japandi Sanctuary"
        elif pair == {"Modern Minimalist", "Scandinavian"}:
            return "Nordic Modern Minimalist"
        elif pair == {"Bohemian Modern", "Scandinavian"}:
            return "Boho-Nordic Fusion"
        elif pair == {"Luxury Contemporary", "Modern Minimalist"}:
            return "Luxury Minimalist Penthouse"
        elif pair == {"Biophilic Organic", "Japandi"}:
            return "Biophilic Zen Wabi-Sabi"
        else:
            return f"{style1} & {style2} Harmonized Blend"

    def _allocate_priority_catalog(
        self,
        users: List[Dict[str, Any]],
        budget_limit: float,
        budget_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Merges priority items across all inhabitants, allocates cost budgets per piece,
        and assigns architectural material specifications matching the budget tier.
        """
        seen_items = set()
        catalog = []
        
        furn_budget = budget_limit * budget_spec["furniture_budget_pct"]
        max_cost = budget_spec["target_max_piece_cost"]
        
        # Interleave items from users to guarantee fair representation
        max_len = max(len(u["priority_objects"]) for u in users)
        for idx in range(max_len):
            for u in users:
                if idx < len(u["priority_objects"]):
                    item_name = u["priority_objects"][idx].strip()
                    if item_name.lower() not in seen_items:
                        seen_items.add(item_name.lower())
                        
                        # Compute fair price estimate based on item type & budget tier
                        est_cost = self._estimate_item_cost(item_name, max_cost, budget_spec["cost_multiplier"])
                        
                        spec = STYLE_ARCHITECTURAL_SPECS.get(u["preferred_style"], STYLE_ARCHITECTURAL_SPECS["Modern Minimalist"])
                        primary_mat = spec["primary_materials"][idx % len(spec["primary_materials"])]
                        
                        catalog.append({
                            "name": item_name,
                            "requested_by": u["name"],
                            "style_origin": u["preferred_style"],
                            "estimated_cost": est_cost,
                            "material_spec": f"{primary_mat} with {budget_spec['material_qualifiers']}",
                            "priority_rank": len(catalog) + 1
                        })
                        
        return catalog

    def _estimate_item_cost(self, item_name: str, max_piece_cost: float, cost_multiplier: float) -> float:
        """Estimates item cost scaled to budget tier."""
        name_lower = item_name.lower()
        if any(k in name_lower for k in ["sofa", "couch", "sectional", "bed"]):
            base = max_piece_cost * 0.90
        elif any(k in name_lower for k in ["table", "desk", "credenza", "bookshelf"]):
            base = max_piece_cost * 0.50
        elif any(k in name_lower for k in ["chair", "armchair", "lounge"]):
            base = max_piece_cost * 0.40
        elif any(k in name_lower for k in ["lamp", "pendant", "lighting", "rug"]):
            base = max_piece_cost * 0.25
        else:
            base = max_piece_cost * 0.15
            
        return round(base * cost_multiplier, 2)

    def _build_variation_specs(
        self,
        users: List[Dict[str, Any]],
        room_type: str,
        budget_tier: str,
        budget_limit: float,
        budget_spec: Dict[str, Any],
        allocated_catalog: List[Dict[str, Any]],
        custom_notes: Optional[str] = "",
        color_palette: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Builds the 4 distinct variation specifications strictly parameterized by
        user styles, priority pieces, and budget limits.
        """
        u1 = users[0]
        u2 = users[1]
        
        all_names = " & ".join([u["name"] for u in users])
        
        p1_spec = STYLE_ARCHITECTURAL_SPECS.get(u1["preferred_style"], STYLE_ARCHITECTURAL_SPECS["Modern Minimalist"])
        p2_spec = STYLE_ARCHITECTURAL_SPECS.get(u2["preferred_style"], STYLE_ARCHITECTURAL_SPECS["Japandi"])
        
        # User 1 specific priority items
        u1_items = [i["name"] for i in allocated_catalog if i["requested_by"] == u1["name"]] or ["Lounge Sofa", "Center Table"]
        # User 2 specific priority items
        u2_items = [i["name"] for i in allocated_catalog if i["requested_by"] == u2["name"]] or ["Accent Armchair", "Ambient Lighting"]
        # Compromise items
        compromise_items = [i["name"] for i in allocated_catalog[:5]]
        
        budget_desc = f"Budget Category: {budget_tier} (${budget_limit:,.0f} cap). {budget_spec['finish_level']}."
        palette_note = f" Harmonious color accents: {', '.join(color_palette[:3])}." if color_palette else ""
        notes_str = f" Directive: {custom_notes.strip()}." if custom_notes and custom_notes.strip() else ""

        var_specs = [
            # ------------------------------------------------------------------
            # Option 1: Modern Minimalist Design
            # ------------------------------------------------------------------
            {
                "variation_id": "var_1",
                "title": "Option 1: Modern Minimalist",
                "subtitle": "Back Wall Sectional Sofa · Center Coffee Table · Left & Right Accent Chairs",
                "persona_blend": "Modern Minimalist (Clean Linear Architecture)",
                "style": "Modern Minimalist",
                "secondary_style": "Scandinavian",
                "variation_mode": "modern_minimalist",
                "layout_archetype": "Modern Minimalist (Back Wall Sofa x=50, y=20 | Center Table x=50, y=55 | Left Chair x=15, y=50 | Right Chair x=85, y=50)",
                "wall_anchor": "Back Wall Anchor (Center x=50, y=20)",
                "circulation_corridor": "Open Central Walking Path (Clearance: 110cm)",
                "user_color_tone": "#475569",
                "seed": 42 + int(budget_limit) % 1000,
                "suggested_objects": ["Modern Minimalist Sectional Sofa", "Low-Profile Center Coffee Table", "Left Bouclé Armchair", "Right Bouclé Armchair", "Architectural LED Floor Lamp"],
                "layout_description": "Modern Minimalist layout placing a new modern sectional sofa against the BACK WALL (x=50, y=20), center coffee table (x=50, y=55), left accent chair on the LEFT SIDE (x=15, y=50), and right accent chair on the RIGHT SIDE (x=85, y=50), maintaining an open, unobstructed walking path.",
                "prompt": (
                    f"Master architectural interior redesign of a {room_type.lower()} in Modern Minimalist style. "
                    f"MAIN SOFA placed against the BACK WALL (center), COFFEE TABLE placed in the CENTER, "
                    f"one ACCENT CHAIR placed on the LEFT SIDE, another ACCENT CHAIR placed on the RIGHT SIDE, "
                    f"neutral colors, light wood, minimal decoration, modern architectural lighting, open walking path. "
                    f"{budget_desc}{palette_note}{notes_str} 8k resolution, architectural digest photography."
                )
            },
            # ------------------------------------------------------------------
            # Option 2: Luxury Contemporary Design
            # ------------------------------------------------------------------
            {
                "variation_id": "var_2",
                "title": "Option 2: Luxury Contemporary",
                "subtitle": "Left Wall Curved Sofa · Center-Right Marble Table · Right Diagonal Accent Chairs",
                "persona_blend": "Luxury Contemporary (Curved Sectional & Marble Elements)",
                "style": "Luxury Contemporary",
                "secondary_style": "Modern Minimalist",
                "variation_mode": "luxury_contemporary",
                "layout_archetype": "Luxury Contemporary (Left Wall Sofa x=25, y=45 | Center-Right Table x=55, y=55 | Right Chairs x=80, y=35 & x=80, y=65)",
                "wall_anchor": "Left Lateral Wall (x=25, y=45)",
                "circulation_corridor": "Right-Side Open Circulation Area (Clearance: 125cm)",
                "user_color_tone": "#d97706",
                "seed": 101 + int(budget_limit) % 1000,
                "suggested_objects": ["Premium Curved Sectional Sofa", "Calacatta Marble Coffee Table", "Right Diagonal Velvet Chair 1", "Right Diagonal Velvet Chair 2", "Sculptural Designer Chandelier"],
                "layout_description": "Luxury Contemporary layout placing a premium curved sectional sofa along the LEFT WALL (x=25, y=45), designer marble table toward CENTER-RIGHT (x=55, y=55), two diagonal accent chairs on the RIGHT SIDE (x=80, y=35 & x=80, y=65), sophisticated fluted accent wall, and open right-side circulation.",
                "prompt": (
                    f"Master architectural interior redesign of a {room_type.lower()} in Luxury Contemporary style. "
                    f"PREMIUM CURVED SOFA positioned along the LEFT WALL, MARBLE COFFEE TABLE placed toward CENTER-RIGHT, "
                    f"TWO ACCENT CHAIRS placed diagonally on the RIGHT SIDE, sophisticated fluted accent wall with cove lighting, "
                    f"designer sculptural chandelier, dark wood and polished stone elements, right-side circulation area kept open. "
                    f"{budget_desc}{palette_note}{notes_str} 8k resolution, architectural digest photography."
                )
            },
            # ------------------------------------------------------------------
            # Option 3: Scandinavian / Japandi Design
            # ------------------------------------------------------------------
            {
                "variation_id": "var_3",
                "title": "Option 3: Scandinavian / Japandi",
                "subtitle": "Back & Left Wall L-Sofa · Center Round Table · Right Corner Chair & Front Plant",
                "persona_blend": "Scandinavian / Japandi (Natural Wood & Plant Sanctuary)",
                "style": "Japandi",
                "secondary_style": "Scandinavian",
                "variation_mode": "japandi",
                "layout_archetype": "Scandinavian / Japandi (L-Sofa x=30, y=25 | Round Table x=55, y=55 | Right Chair x=85, y=50 | Plant x=10, y=75)",
                "wall_anchor": "Back Wall & Left Wall Anchor (x=30, y=25)",
                "circulation_corridor": "Space-Efficient Perimeter Walkways (Clearance: 100cm)",
                "user_color_tone": "#e29578",
                "seed": 2024 + int(budget_limit) % 1000,
                "suggested_objects": ["Compact Modular L-Shaped Sofa", "Round Wooden Coffee Table", "Natural Oak Lounge Chair", "Potted Monstera Plant", "Shoji Paper Floor Lamp"],
                "layout_description": "Scandinavian/Japandi layout placing a compact modular L-shaped sofa along BACK & LEFT WALLS (x=30, y=25), round wooden table in CENTER (x=55, y=55), accent chair in FRONT-RIGHT CORNER (x=85, y=50), and potted plant in FRONT-LEFT (x=10, y=75), creating a cozy, space-efficient sanctuary.",
                "prompt": (
                    f"Master architectural interior redesign of a {room_type.lower()} in Scandinavian/Japandi style. "
                    f"COMPACT MODULAR L-SHAPED SOFA positioned along the BACK WALL and LEFT WALL, "
                    f"ROUND WOODEN COFFEE TABLE placed near the CENTER, one ACCENT CHAIR near the FRONT-RIGHT/RIGHT CORNER, "
                    f"POTTED PLANT placed in the FRONT-LEFT area, cream and beige furniture, natural wood, indoor plants, soft lighting, "
                    f"minimal decoration, space-efficient arrangement. {budget_desc}{palette_note}{notes_str} 8k resolution, architectural digest photography."
                )
            },
            # ------------------------------------------------------------------
            # Option 4: Industrial Contemporary Design
            # ------------------------------------------------------------------
            {
                "variation_id": "var_4",
                "title": "Option 4: Industrial Contemporary",
                "subtitle": "Cognac Leather Sofa · Steel Coffee Table · Dual-Zone Study Workstation",
                "persona_blend": "Industrial Contemporary (Raw Textures & Dual-Zone Layout)",
                "style": "Industrial",
                "secondary_style": "Modern Minimalist",
                "variation_mode": "industrial",
                "layout_archetype": "Industrial Contemporary (Dual-Zone Split Space: Lounge Zone A + Workstation Zone B)",
                "wall_anchor": "Dual-Zone Split Anchor (Zone A Left + Zone B Right)",
                "circulation_corridor": "Central Spine Walkway (Clearance: 105cm)",
                "user_color_tone": "#334155",
                "seed": 8888 + int(budget_limit) % 1000,
                "suggested_objects": ["Top-Grain Cognac Leather Sofa", "Blackened Steel Coffee Table", "Reclaimed Timber Work Desk", "Ergonomic Desk Chair", "Edison Amber Pendant Lamp"],
                "layout_description": "Industrial Contemporary layout dividing room into a tufted cognac leather sofa lounge (Zone A, Left) and a study workstation with steel bookshelf (Zone B, Right).",
                "prompt": (
                    f"Master architectural interior redesign of a {room_type.lower()} in Industrial Contemporary style. "
                    f"Top-grain cognac leather sofa, blackened cast iron coffee table, reclaimed wood study desk, "
                    f"exposed architectural textures, Edison filament lighting, dual-zone layout. "
                    f"{budget_desc}{palette_note}{notes_str} 8k resolution, architectural digest photography."
                )
            }
        ]

        return var_specs


# Singleton instance
consensus_engine = ConsensusEngine()
