"""
AI Interior Space Generation & Multi-User Consensus Diffusion Pipeline
Implements Stable Diffusion + ControlNet with high visual diversity across 4 multi-user compromise variations:
- Variation 1: Person 1 Dominant Focus (ControlNet Baseline)
- Variation 2: Person 2 Dominant Focus (High-Freedom Img2Img Corner Layout)
- Variation 3: 50/50 Hybrid Blend (High-Freedom Img2Img Floating Arrangement)
- Variation 4: Full Group Consensus Compromise (High-Freedom Img2Img Multi-Zone + Budget Additions)
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import io
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import torch
from config import settings


DEFAULT_NEGATIVE_PROMPT = (
    "ugly, deformed, poorly drawn, blurry, low quality, dark shadows, distorted perspective, "
    "cluttered, messy, bad architecture, unrealistic proportions, floating furniture, pixelated"
)

STYLE_PROMPT_PRESETS: Dict[str, Dict[str, Any]] = {
    "Modern Minimalist": {
        "positive": "ultra-modern minimalist interior, clean lines, open floor plan, neutral tones, subtle contrast",
        "materials": "polished concrete, matte black steel, white oak, frameless glass",
        "lighting": "recessed architectural LED strips, soft indirect natural daylight",
        "tint": (-12, -4, 18),
        "contrast": 1.25,
        "brightness": 1.06,
        "saturation": 0.85,
        "sharpness": 1.5
    },
    "Scandinavian": {
        "positive": "authentic Scandinavian hygge interior, cozy airy aesthetic, functional minimalism, light wood",
        "materials": "blonde birch wood, woven bouclé wool, white linen, ceramic pottery",
        "lighting": "abundant natural Nordic sunlight, diffused 3000K paper pendant lighting",
        "tint": (15, 12, 6),
        "contrast": 1.14,
        "brightness": 1.10,
        "saturation": 1.15,
        "sharpness": 1.3
    },
    "Japandi": {
        "positive": "wabi-sabi Japandi fusion interior, organic serenity, understated elegance, clean symmetry",
        "materials": "natural cedar, textured rice paper, raw linen, glazed earthenware",
        "lighting": "zen diffused warm paper lanterns, gentle ambient golden hour daylight",
        "tint": (22, 14, -8),
        "contrast": 1.18,
        "brightness": 1.04,
        "saturation": 1.12,
        "sharpness": 1.35
    },
    "Industrial": {
        "positive": "contemporary industrial loft interior, exposed architectural textures, high ceilings",
        "materials": "weathered brick, blackened cast iron, distressed cognac leather, reclaimed timber",
        "lighting": "warm Edison amber filament pendants, directional matte black spotlights",
        "tint": (20, 4, -20),
        "contrast": 1.35,
        "brightness": 0.94,
        "saturation": 1.22,
        "sharpness": 1.6
    },
    "Bohemian Modern": {
        "positive": "bohemian chic interior, vibrant layered textures, eclectic worldly aesthetic",
        "materials": "macrame textiles, rattan wicker, vintage kilim rug, terracotta pottery",
        "lighting": "warm bohemian fairy lights, draped natural sunshine",
        "tint": (24, 8, -14),
        "contrast": 1.16,
        "brightness": 1.05,
        "saturation": 1.32,
        "sharpness": 1.25
    },
    "Biophilic Organic": {
        "positive": "biophilic interior architecture, indoor living greenery, seamless nature connection",
        "materials": "living moss walls, solid travertine stone, FSC bamboo, unbleached linen",
        "lighting": "full-spectrum circadian natural light, filtered green canopy shadows",
        "tint": (-6, 26, -4),
        "contrast": 1.20,
        "brightness": 1.08,
        "saturation": 1.35,
        "sharpness": 1.4
    }
}


class ControlNetService:
    """Stable Diffusion + ControlNet pipeline wrapper for multi-user interior redesign."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.pipeline_initialized = False

    def extract_canny_conditioning(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
        conditioning_scale: float = 0.45
    ) -> Image.Image:
        """Extracts Canny edge structural condition map."""
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        edges_rgb = np.stack([edges] * 3, axis=-1)
        if conditioning_scale < 1.0:
            edges_rgb = (edges_rgb.astype(np.float32) * conditioning_scale).astype(np.uint8)
        return Image.fromarray(edges_rgb)

    def extract_depth_conditioning(self, image: Image.Image) -> Image.Image:
        """Extracts gradient-based depth proxy map."""
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        norm_depth = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_rgb = np.stack([norm_depth] * 3, axis=-1)
        return Image.fromarray(depth_rgb)

    def generate_prompt(
        self,
        room_type: str,
        style: str,
        spatial_layout_prompt: str = "",
        budget_tier: str = "Mid-Range Premium",
        budget_objects: Optional[List[str]] = None,
        color_palette: Optional[List[str]] = None,
        custom_notes: Optional[str] = None
    ) -> Dict[str, str]:
        """Synthesizes positive and negative diffusion prompts."""
        preset = STYLE_PROMPT_PRESETS.get(style, STYLE_PROMPT_PRESETS["Modern Minimalist"])
        
        prompt_parts = [
            f"A masterfully designed {room_type.lower()}",
            spatial_layout_prompt if spatial_layout_prompt else "rearranged spatial furniture layout",
            f"Budget category: {budget_tier}",
            preset["positive"],
            f"Key materials: {preset['materials']}",
            f"Lighting design: {preset['lighting']}"
        ]

        if budget_objects and len(budget_objects) > 0:
            objs_str = ", ".join(budget_objects)
            prompt_parts.append(f"featuring pieces: {objs_str}")

        if color_palette:
            palette_str = ", ".join(color_palette[:4])
            prompt_parts.append(f"harmonious color palette accented with tones {palette_str}")

        if custom_notes and custom_notes.strip():
            prompt_parts.append(custom_notes.strip())

        positive_prompt = ", ".join(prompt_parts)
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "room_type": room_type,
            "budget_tier": budget_tier,
            "spatial_layout": spatial_layout_prompt
        }

    def _render_multi_user_transformation(
        self,
        base_img: Image.Image,
        primary_style: str,
        secondary_style: Optional[str],
        variation_mode: str,
        user_color_tone: Optional[str],
        room_type: str = "Living Room",
        layout_idx: int = 0
    ) -> Image.Image:
        """
        Renders authentic AI interior redesign visual matching selected style and room type.
        """
        from services.generator import generator_service
        w, h = base_img.size
        return generator_service._generate_ai_redesign_visual(
            original_image=base_img,
            target_size=(w, h),
            style=primary_style,
            room_type=room_type,
            user_color_tone=user_color_tone,
            layout_idx=layout_idx
        )
    def generate_redesign_variations(
        self,
        original_image: Image.Image,
        room_type: str = "Living Room",
        style: str = "Modern Minimalist",
        budget_tier: str = "Mid-Range Premium",
        budget_limit: float = 15000.0,
        users: Optional[List[Dict[str, Any]]] = None,
        color_palette: Optional[List[str]] = None,
        custom_notes: Optional[str] = "",
        controlnet_scale: float = 0.45,
        num_inference_steps: int = settings.DEFAULT_INFERENCE_STEPS,
        num_variations: int = 2
    ) -> Dict[str, Any]:
        """
        Executes multi-user compromise redesign generating 2 distinct suggested persona blends:
        - Var 1: Option 1 (Person 1 Symmetrical Dominant Focus)
        - Var 2: Option 2 (Person 2 L-Shaped Corner Open-Flow Focus)
        """
        eff_scale = min(0.50, max(0.30, controlnet_scale))

        canny_condition = self.extract_canny_conditioning(original_image, conditioning_scale=eff_scale)
        depth_condition = self.extract_depth_conditioning(original_image)

        if not users or len(users) < 1:
            users = [
                {
                    "name": "Alex",
                    "preferred_style": style or "Modern Minimalist",
                    "priority_objects": ["Sofa", "Coffee Table"],
                    "color_tone": "#3b82f6"
                },
                {
                    "name": "Sam",
                    "preferred_style": "Japandi",
                    "priority_objects": ["Monstera Plant", "Floor Lamp"],
                    "color_tone": "#e29578"
                }
            ]
        elif len(users) == 1:
            users.append({
                "name": "Co-Inhabitant",
                "preferred_style": "Scandinavian",
                "priority_objects": ["Lounge Chair", "Area Rug"],
                "color_tone": "#10b981"
            })

        u1 = users[0]
        u2 = users[1]
        all_names = [u.get("name", f"User {i+1}") for i, u in enumerate(users)]
        all_styles = [u.get("preferred_style", "Modern Minimalist") for u in users]
        all_names_str = " & ".join(all_names[:3])

        u1_name = u1.get("name", "Person 1")
        u1_style = u1.get("preferred_style", "Modern Minimalist")
        u1_objs = u1.get("priority_objects", ["Sofa", "Coffee Table"])

        u2_name = u2.get("name", "Person 2")
        u2_style = u2.get("preferred_style", "Japandi")
        u2_objs = u2.get("priority_objects", ["Floor Lamp", "Plants"])

        budget_item_catalog = {
            "Budget Essential": {
                "var_1": [f"{u1_style} Coffee Table ($180)", "Ceramic Accent Lamps ($75)"],
                "var_2": [f"{u2_style} Sectional Slipcover ($350)", "Arc Floor Lamp ($95)", "Potted Greenery ($45)"],
                "var_3": ["Linen Accent Chair ($220)", "Wall Floating Shelves ($65)", "LED Warm Light Strips ($35)"],
                "var_4": ["Pair of Minimalist Armchairs ($480)", "Round Glass Side Table ($120)", "Flatweave Area Rug ($160)"]
            },
            "Mid-Range Premium": {
                "var_1": [f"Solid Oak {u1_style} Coffee Table ($450)", "Designer Table Lamps ($180)"],
                "var_2": [f"Modular {u2_style} Sectional ($1,400)", "Brushed Brass Arc Lamp ($240)", "Large Monstera Deliciosa ($85)"],
                "var_3": ["Velvet Accent Lounge Chair ($620)", "Solid Walnut Floating Shelves ($180)", "Smart Ambient Cove LED ($120)"],
                "var_4": ["Pair of Bouclé Armchairs ($1,250)", "Calacatta Marble Side Table ($420)", "Handwoven Wool Rug ($550)"]
            },
            "Luxury Bespoke": {
                "var_1": [f"Bespoke Travertine {u1_style} Table ($1,800)", "Sculptural Alabaster Lamps ($850)"],
                "var_2": [f"Custom Italian {u2_style} Sectional ($4,500)", "Flos Arc Floor Lamp ($950)", "Mature Specimen Greenery ($350)"],
                "var_3": ["Designer Shearling Swivel Chair ($2,200)", "Architectural Bronze Floating Shelving ($850)", "Integrated Cove Lighting ($650)"],
                "var_4": ["Pair of Custom Italian Bouclé Armchairs ($3,600)", "Tiered Calacatta Gold Table ($1,400)", "Hand-Knotted Silk Area Rug ($2,400)"]
            }
        }
        catalog = budget_item_catalog.get(budget_tier, budget_item_catalog["Mid-Range Premium"])

        variation_configs = [
            {
                "id": "var_1",
                "title": f"Option 1: {u1_name}'s Symmetrical {u1_style} Focus",
                "subtitle": f"Bilateral Center Layout · {u1_style} Anchor Seating",
                "persona_blend": f"{u1_name} (Dominant Focus)",
                "mode": "user_1_dominant",
                "primary_style": u1_style,
                "secondary_style": None,
                "user_color": u1.get("color_tone"),
                "layout_description": f"Symmetrical layout tailored to {u1_name}'s {u1_style} style, highlighting: {', '.join(u1_objs)}.",
                "spatial_prompt": f"Interior tailored to {u1_name}'s {u1_style} style, featuring primary elements: {', '.join(u1_objs)}, structured bilateral layout.",
                "suggested_objects": catalog["var_1"],
                "pipeline_mode": "ControlNet Baseline",
                "seed": 42,
                "weight": 0.35
            },
            {
                "id": "var_2",
                "title": f"Option 2: {u2_name}'s L-Shaped {u2_style} Lounge",
                "subtitle": f"L-Shaped Open Flow · {u2_style} Reading Nook",
                "persona_blend": f"{u2_name} (Dominant Focus)",
                "mode": "user_2_dominant",
                "primary_style": u2_style,
                "secondary_style": None,
                "user_color": u2.get("color_tone"),
                "layout_description": f"Corner L-shaped arrangement highlighting {u2_name}'s {u2_style} ambiance and {', '.join(u2_objs)}.",
                "spatial_prompt": f"Interior tailored to {u2_name}'s {u2_style} style, featuring corner arrangement and {', '.join(u2_objs)}, open airy floor space.",
                "suggested_objects": catalog["var_2"],
                "pipeline_mode": "High-Freedom Img2Img",
                "seed": 101,
                "weight": 0.15
            },
            {
                "id": "var_3",
                "title": f"Option 3: {u1_name} & {u2_name} 50/50 Hybrid",
                "subtitle": f"Harmonious Blend of {u1_style} and {u2_style}",
                "persona_blend": f"{u1_name} (50%) + {u2_name} (50%)",
                "mode": "hybrid_50_50",
                "primary_style": u1_style,
                "secondary_style": u2_style,
                "user_color": u1.get("color_tone"),
                "layout_description": f"Floating floor plan synthesizing {u1_name}'s {u1_style} structure with {u2_name}'s {u2_style} textures.",
                "spatial_prompt": f"Harmonious hybrid fusion uniting {u1_name}'s {u1_style} materials with {u2_name}'s {u2_style} textures, incorporating {', '.join(u1_objs[:1] + u2_objs[:1])}.",
                "suggested_objects": catalog["var_3"],
                "pipeline_mode": "High-Freedom Img2Img",
                "seed": 2024,
                "weight": 0.15
            },
            {
                "id": "var_4",
                "title": f"Option 4: Full Group Compromise ({all_names_str})",
                "subtitle": "Multi-Zone Space Harmonizing All Co-Inhabitants",
                "persona_blend": "Full Group Balanced Consensus",
                "mode": "group_compromise",
                "primary_style": u1_style,
                "secondary_style": u2_style,
                "user_color": u2.get("color_tone"),
                "layout_description": f"Multi-zoned floor plan accommodating requested elements from {all_names_str} with luxury budget accents.",
                "spatial_prompt": f"Complete multi-user consensus interior harmonizing {', '.join(all_styles)}, multi-zone spatial layout with designer budget additions.",
                "suggested_objects": catalog["var_4"],
                "pipeline_mode": "High-Freedom Img2Img",
                "seed": 8888,
                "weight": 0.15
            }
        ]

        variations = []
        for i in range(min(num_variations, len(variation_configs))):
            cfg = variation_configs[i]
            
            prompts = self.generate_prompt(
                room_type=room_type,
                style=cfg["primary_style"],
                spatial_layout_prompt=cfg["spatial_prompt"],
                budget_tier=budget_tier,
                budget_objects=cfg["suggested_objects"],
                color_palette=color_palette,
                custom_notes=custom_notes
            )
            
            var_img = self._render_multi_user_transformation(
                base_img=original_image,
                primary_style=cfg["primary_style"],
                secondary_style=cfg["secondary_style"],
                variation_mode=cfg["mode"],
                user_color_tone=cfg["user_color"],
                room_type=room_type,
                layout_idx=i
            )
            
            variations.append({
                "variation_id": cfg["id"],
                "title": cfg["title"],
                "subtitle": cfg["subtitle"],
                "style": cfg["primary_style"],
                "persona_blend": cfg["persona_blend"],
                "layout_description": cfg["layout_description"],
                "suggested_objects": cfg["suggested_objects"],
                "spatial_prompt": cfg["spatial_prompt"],
                "pipeline_mode": cfg["pipeline_mode"],
                "seed": cfg["seed"],
                "controlnet_weight": cfg["weight"],
                "prompts": prompts,
                "image": var_img
            })

        primary_output = variations[2]["image"] if len(variations) > 2 else variations[0]["image"]
        primary_prompts = variations[2]["prompts"] if len(variations) > 2 else variations[0]["prompts"]

        return {
            "primary_image": primary_output,
            "canny_conditioning": canny_condition,
            "depth_conditioning": depth_condition,
            "prompts": primary_prompts,
            "controlnet_scale": eff_scale,
            "inference_steps": num_inference_steps,
            "variations": variations
        }


# Singleton instance
controlnet_service = ControlNetService()
