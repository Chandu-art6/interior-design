"""
Vision LLM Analyzer Module
Analyzes room photos using Multimodal Vision AI (Gemini Vision / GPT-4o / Local Vision Heuristics)
and outputs structured JSON identifying items to remove, items to keep, purchase suggestions,
and an optimized inpainting prompt.
"""
import os
import json
import base64
import io
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from config import settings
from services.yolo_service import yolo_service


class RoomAnalyzer:
    """
    Vision Intelligence Analyzer for Room Redesign & Inpainting.
    Synthesizes visual room analysis with structural preservation constraints.
    """

    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", getattr(settings, "OPENROUTER_MODEL", "openrouter/free"))
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    def analyze_room_for_inpainting(
        self,
        image: Image.Image,
        room_type: str = "Living Room",
        target_style: str = "Modern Minimalist",
        budget_tier: str = "Mid-Range Premium",
        budget_limit: float = 15000.0,
        custom_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint: analyzes room photo and returns structured JSON with:
        - items_to_remove
        - items_to_keep
        - purchase_suggestions
        - inpainting_prompt
        - style_rationale
        """
        # 1. Try OpenRouter Vision Model (using user provided free/configured key)
        if self.openrouter_api_key:
            try:
                openrouter_res = self._call_openrouter_vision(
                    image, room_type, target_style, budget_tier, budget_limit, custom_notes
                )
                if openrouter_res:
                    return openrouter_res
            except Exception as e:
                print(f"[Analyzer] OpenRouter Vision call failed, falling back to Gemini/Local: {e}")

        # 2. Try Gemini Vision if API key available
        if self.gemini_api_key:
            try:
                gemini_res = self._call_gemini_vision(image, room_type, target_style, budget_tier, budget_limit, custom_notes)
                if gemini_res:
                    return gemini_res
            except Exception as e:
                print(f"[Analyzer] Gemini Vision API call failed, falling back to OpenAI/Local: {e}")

        # 3. Try OpenAI GPT-4o Vision if API key available
        if self.openai_api_key:
            try:
                openai_res = self._call_openai_vision(image, room_type, target_style, budget_tier, budget_limit, custom_notes)
                if openai_res:
                    return openai_res
            except Exception as e:
                print(f"[Analyzer] OpenAI GPT-4o Vision call failed, falling back to Local Vision Engine: {e}")

        # 4. Fast, Deterministic Local Vision & Object Reasoning Engine
        return self._local_vision_analysis(image, room_type, target_style, budget_tier, budget_limit, custom_notes)

    def _call_openrouter_vision(
        self,
        image: Image.Image,
        room_type: str,
        target_style: str,
        budget_tier: str,
        budget_limit: float,
        custom_notes: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Invokes OpenRouter Vision / Multimodal model (e.g. nvidia/nemotron-nano-12b-v2-vl:free or openrouter/free)."""
        import requests
        import json

        buffered = io.BytesIO()
        image.resize((800, 800)).save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = self._build_vision_prompt(room_type, target_style, budget_tier, budget_limit, custom_notes)
        models_to_try = [self.openrouter_model] if self.openrouter_model else ["nvidia/nemotron-nano-12b-v2-vl:free"]

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Interior Redesign"
        }

        for model_id in models_to_try:
            try:
                payload = {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt + "\nRespond ONLY with valid RFC8259 JSON without markdown fences."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                                }
                            ]
                        }
                    ]
                }
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=2.0)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    parsed = json.loads(content.strip())
                    if isinstance(parsed, dict) and "inpainting_prompt" in parsed:
                        return parsed
            except Exception:
                pass

        return None


    def _call_gemini_vision(
        self,
        image: Image.Image,
        room_type: str,
        target_style: str,
        budget_tier: str,
        budget_limit: float,
        custom_notes: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Invokes Google Gemini 1.5 Pro / Flash Vision."""
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = self._build_vision_prompt(room_type, target_style, budget_tier, budget_limit, custom_notes)
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        
        # Clean JSON markdown fences if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())

    def _call_openai_vision(
        self,
        image: Image.Image,
        room_type: str,
        target_style: str,
        budget_tier: str,
        budget_limit: float,
        custom_notes: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Invokes OpenAI GPT-4o Vision."""
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_api_key)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = self._build_vision_prompt(room_type, target_style, budget_tier, budget_limit, custom_notes)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _local_vision_analysis(
        self,
        image: Image.Image,
        room_type: str,
        target_style: str,
        budget_tier: str,
        budget_limit: float,
        custom_notes: Optional[str]
    ) -> Dict[str, Any]:
        """
        Robust Local Computer Vision Analyzer.
        Runs YOLO object detection, examines spatial coordinates, and builds targeted replacement lists.
        """
        # Save temp image for YOLO detection
        temp_path = settings.UPLOAD_DIR / "temp_analyze.jpg"
        image.save(temp_path, quality=90)
        yolo_res = yolo_service.detect_furniture(temp_path)
        detections = yolo_res.get("detections", [])

        # Catalog of replacement strategies per style
        style_materials = {
            "Modern Minimalist": "ultra-clean matte black metal, white oak, crisp neutral linen, low-profile silhouette",
            "Scandinavian": "blonde ash wood, bouclé texture, white wool rug, diffused organic paper lighting",
            "Japandi": "slatted cedar wood, wabi-sabi ceramic pottery, textured oatmeal linen, bamboo accents",
            "Industrial": "distressed cognac leather, blackened cast iron frame, reclaimed timber, Edison amber fixtures",
            "Bohemian": "rattan wicker weave, layered vintage kilim rug, macramé textures, leafy terracotta planters",
            "Luxury Contemporary": "calacatta gold marble, brushed brass accents, rich velvet upholstery, architectural cove lighting",
            "Biophilic": "organic teak, living moss wall accents, textured linen, terracotta planters, natural hemp"
        }
        material_desc = style_materials.get(target_style, style_materials["Modern Minimalist"])

        items_to_remove = []
        items_to_keep = [
            {"name": "Structural Walls & Ceiling", "reason": "Load-bearing architectural envelope and ceiling boundary.", "category": "Architecture"},
            {"name": "Window Openings", "reason": "Natural daylight ingress vectors and outdoor sightlines.", "category": "Architecture"},
            {"name": "Flooring Substrate", "reason": "Base structural room geometry and perspective vanishing lines.", "category": "Architecture"}
        ]

        # Classify detected furniture into replaceable vs preserved
        for det in detections:
            label = det.get("label", "furniture")
            bbox = det.get("bbox_normalized", [0.2, 0.4, 0.8, 0.8])
            
            if label.lower() in ["couch", "chair", "sofa", "dining table", "coffee table", "bed", "rug", "desk", "lamp"]:
                items_to_remove.append({
                    "name": f"Existing {label.title()}",
                    "reason": f"Replace with {target_style} styled ergonomic piece using {material_desc.split(',')[0]}.",
                    "category": "Furniture",
                    "bbox": bbox
                })
            elif label.lower() in ["tv", "refrigerator", "wardrobe"]:
                items_to_keep.append({
                    "name": f"Existing {label.upper() if label == 'tv' else label.title()}",
                    "reason": "Keep in current functional position to preserve utility wiring and wall anchors.",
                    "category": "Appliance / Fixture"
                })

        if not items_to_remove:
            items_to_remove.append({
                "name": "Central Seating & Coffee Table Zone",
                "reason": f"Upgrade central living zone to {target_style} aesthetic.",
                "category": "Furniture",
                "bbox": [0.15, 0.45, 0.85, 0.90]
            })

        # Curate purchase suggestions matching style & budget tier
        cost_multiplier = {"Budget Essential": 0.5, "Mid-Range Premium": 1.0, "Luxury Bespoke": 2.5}.get(budget_tier, 1.0)
        
        purchase_catalog = {
            "Modern Minimalist": [
                {"name": "Low-Profile Modular Sectional in Dove Grey", "category": "Seating", "cost": int(1600 * cost_multiplier), "material": "High-density foam, stain-resistant linen", "rationale": "Maintains open sightlines and low center of gravity."},
                {"name": "Solid White Oak Floating Coffee Table", "category": "Tables", "cost": int(420 * cost_multiplier), "material": "Solid White Oak", "rationale": "Provides warm organic contrast against neutral floors."},
                {"name": "Matte Black Bilateral Floor Arc Lamp", "category": "Lighting", "cost": int(220 * cost_multiplier), "material": "Powder-coated steel", "rationale": "Directs 3000K warm task light over conversation zone."}
            ],
            "Scandinavian": [
                {"name": "Bouclé Upholstered Curved 3-Seater Sofa", "category": "Seating", "cost": int(1450 * cost_multiplier), "material": "Textured Bouclé Wool", "rationale": "Adds tactile hygge warmth and acoustic dampening."},
                {"name": "Blonde Ash Nesting Center Tables", "category": "Tables", "cost": int(380 * cost_multiplier), "material": "FSC-certified Ash", "rationale": "Flexible arrangement that preserves walking corridors."},
                {"name": "Handwoven Flatweave Geometric Wool Area Rug", "category": "Rugs", "cost": int(320 * cost_multiplier), "material": "100% New Zealand Wool", "rationale": "Anchors the seating group and absorbs floor reflections."}
            ],
            "Japandi": [
                {"name": "Solid Hinoki Cypress Frame Low Sofa with Oatmeal Cushions", "category": "Seating", "cost": int(1850 * cost_multiplier), "material": "Cypress Wood & Organic Cotton", "rationale": "Emphasizes wabi-sabi simplicity and grounding posture."},
                {"name": "Sculptural Travertine Stone Low Coffee Table", "category": "Tables", "cost": int(680 * cost_multiplier), "material": "Honed Travertine", "rationale": "Natural earthen focal point with zero visual clutter."},
                {"name": "Handmade Washi Rice Paper Pendant / Floor Lamp", "category": "Lighting", "cost": int(190 * cost_multiplier), "material": "Mulberry Paper & Bamboo", "rationale": "Creates soft, shadowless 2700K diffused glow."}
            ],
            "Industrial": [
                {"name": "Distressed Cognac Top-Grain Leather Sectional", "category": "Seating", "cost": int(2100 * cost_multiplier), "material": "Full-grain Italian Leather", "rationale": "Rich patina and timeless loft centerpiece."},
                {"name": "Blackened Cast Iron & Reclaimed Timber Cart Table", "category": "Tables", "cost": int(520 * cost_multiplier), "material": "Iron & Reclaimed Pine", "rationale": "Adds structural weight and architectural texture."},
                {"name": "Articulated Steel Task Floor Lamp with Amber Edison Bulb", "category": "Lighting", "cost": int(175 * cost_multiplier), "material": "Gunmetal Steel", "rationale": "Directional accent illumination."}
            ]
        }
        suggestions_raw = purchase_catalog.get(target_style, purchase_catalog["Modern Minimalist"])
        suggestions = [
            {
                "name": item["name"],
                "category": item["category"],
                "estimated_cost": f"₹{item['cost']:,}",
                "material": item["material"],
                "placement_rationale": item["rationale"],
                "style_match": target_style
            }
            for item in suggestions_raw
        ]

        # Synthesize Inpainting Prompt
        prompt_items = ", ".join([item["name"] for item in suggestions_raw[:2]])
        inpainting_prompt = (
            f"photorealistic {target_style.lower()} interior redesign replacing central area with {prompt_items}, "
            f"crafted from {material_desc}, perfectly integrated with room lighting, 8k resolution, architectural digest quality"
        )
        if custom_notes:
            inpainting_prompt += f", {custom_notes}"

        return {
            "items_to_remove": items_to_remove,
            "items_to_keep": items_to_keep,
            "purchase_suggestions": suggestions,
            "inpainting_prompt": inpainting_prompt,
            "style_rationale": (
                f"Transformation applies {target_style} principles by replacing outdated focal elements with {material_desc}. "
                f"All structural walls, window daylight portals, and electrical fixtures remain strictly preserved."
            )
        }

    def _build_vision_prompt(
        self,
        room_type: str,
        target_style: str,
        budget_tier: str,
        budget_limit: float,
        custom_notes: Optional[str]
    ) -> str:
        """Constructs strict JSON prompt for Multimodal LLMs."""
        return f"""
You are an expert interior architect and computer vision space analyzer.
Analyze this {room_type} photograph for a targeted AI inpainting and redesign in the '{target_style}' style with a '{budget_tier}' budget (approx ₹{budget_limit:,}).

Return ONLY a valid, raw JSON object with this exact structure:
{{
  "items_to_remove": [
    {{
      "name": "Old Dark Fabric Sofa",
      "reason": "Heavy outdated profile conflicting with {target_style} aesthetic.",
      "category": "Furniture",
      "bbox": [ymin, xmin, ymax, xmax] (normalized 0.0 to 1.0)
    }}
  ],
  "items_to_keep": [
    {{
      "name": "Large Bay Window",
      "reason": "Preserves natural daylight ingress and room structural geometry.",
      "category": "Architecture"
    }}
  ],
  "purchase_suggestions": [
    {{
      "name": "Specific product recommendation",
      "category": "Seating / Lighting / Table / Decor",
      "estimated_cost": "$X,XXX",
      "material": "Wood / Fabric / Metal type",
      "placement_rationale": "Why and where to place it",
      "style_match": "{target_style}"
    }}
  ],
  "inpainting_prompt": "Detailed photorealistic diffusion prompt specifying the new replacement furniture, materials, and lighting.",
  "style_rationale": "Brief architectural explanation of the transformation."
}}
Do NOT wrap in markdown fences or add explanatory text outside JSON.
"""


# Singleton instance
analyzer_service = RoomAnalyzer()
