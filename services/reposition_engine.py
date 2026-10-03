"""
Spatial Object Repositioning & Furniture Reorganization Engine
Rearranges detected objects and furniture inside the uploaded room photo into distinct architectural layouts:
- Option 1: Bilateral Center Symmetrical Layout (Anchor Seating Centered along Back Wall)
- Option 2: Corner L-Shaped / Open-Flow Arrangement (Seating Shifted to Lateral Wall, Opening Diagonal Pathway)
- Option 3: Floating Central Island (Detached 360-degree perimeter circulation)
- Option 4: Dual-Zone Split Space (Zone A Seating + Zone B Workstation/Reading Nook)
"""
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import cv2

from config import settings


class SpatialRepositionEngine:
    """
    Extracts furniture objects from the uploaded room image, inpaints the vacated space,
    and repositions furniture into new spatial coordinates with perspective scaling and contact shadows.
    """

    def inpaint_clean_room_background(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        inpaint_radius: int = 21
    ) -> Image.Image:
        """
        Creates a clean room canvas by inpainting regions occupied by detected furniture.
        Uses Telea / Navier-Stokes inpainting with morphological dilation to cleanly fill floor and walls.
        """
        w, h = image.size
        img_np = np.array(image.convert("RGB"))
        mask = np.zeros((h, w), dtype=np.uint8)

        # Build mask for all detected furniture items
        for det in detections:
            bbox = det.get("bbox") or det.get("bbox_normalized")
            if not bbox or len(bbox) != 4:
                continue

            if all(0.0 <= v <= 1.0 for v in bbox):
                x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
                x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
            else:
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))

            if x2 > x1 and y2 > y1:
                # Fill mask with slight margin
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # If no furniture detected, create a synthetic lower-half floor mask for clean room testing
        if np.sum(mask) == 0:
            cv2.rectangle(mask, (int(w * 0.2), int(h * 0.55)), (int(w * 0.8), int(h * 0.85)), 255, -1)

        # Dilate mask slightly to cover object boundary edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)

        # Fast Telea inpainting
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(img_bgr, dilated_mask, inpaint_radius, cv2.INPAINT_TELEA)

        # Blend smooth texture back
        clean_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        clean_pil = Image.fromarray(clean_rgb)

        return clean_pil

    def extract_furniture_sprites(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts cropped furniture sprites with smooth feathered alpha transparency.
        """
        w, h = image.size
        img_np = np.array(image.convert("RGB"))
        sprites = []

        for det in detections:
            bbox = det.get("bbox") or det.get("bbox_normalized")
            if not bbox or len(bbox) != 4:
                continue

            if all(0.0 <= v <= 1.0 for v in bbox):
                x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
                x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
            else:
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
            bw, bh = x2 - x1, y2 - y1

            if bw < 20 or bh < 20:
                continue

            crop = img_np[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop).convert("RGBA")

            # Create soft feathered alpha mask
            alpha_mask = Image.new("L", (bw, bh), 0)
            draw = ImageDraw.Draw(alpha_mask)
            inset = min(8, bw // 6, bh // 6)
            draw.rounded_rectangle([inset, inset, bw - inset, bh - inset], radius=max(4, inset), fill=255)
            alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=3))

            crop_pil.putalpha(alpha_mask)

            sprites.append({
                "category": det.get("category", "Seating"),
                "class_name": det.get("class_name", "furniture"),
                "orig_bbox": [x1, y1, x2, y2],
                "orig_center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "orig_size": (bw, bh),
                "image": crop_pil
            })

        return sprites

    def reposition_room(
        self,
        image: Image.Image,
        detections: Optional[List[Dict[str, Any]]] = None,
        layout_idx: int = 0,
        style: str = "Modern Minimalist",
        room_type: str = "Living Room",
        user_color_tone: Optional[str] = None
    ) -> Image.Image:
        """
        Main entrypoint:
        1. Inpaints clean background room from uploaded image.
        2. Repositions furniture to layout-specific target coordinates.
        3. Renders ground contact shadows and composites repositioned items.
        """
        w, h = image.size
        dets = detections or []

        # 1. Generate clean empty background canvas
        clean_bg = self.inpaint_clean_room_background(image, dets)

        # 2. Extract existing furniture sprites
        sprites = self.extract_furniture_sprites(image, dets)

        # 3. Target Layout Anchor Coordinates
        # Layout 0: Center Symmetrical (Sofa center, table center, flanked lamps)
        # Layout 1: L-Shaped Corner (Sofa West wall, table West corner, East reading corner)
        # Layout 2: Floating Central Island (Sofa center-floating, armchairs left/right)
        # Layout 3: Dual-Zone Multi-Purpose (Zone A left, Zone B right)
        if layout_idx == 0:
            target_slots = [
                {"role": "seating", "center": (0.50, 0.65), "size_scale": 1.05, "name": "Center Main Sofa"},
                {"role": "surface", "center": (0.50, 0.82), "size_scale": 0.95, "name": "Center Coffee Table"},
                {"role": "decor",   "center": (0.18, 0.60), "size_scale": 0.90, "name": "Left Botanical Accent"},
                {"role": "lighting","center": (0.82, 0.60), "size_scale": 0.90, "name": "Right Flank Lamp"},
                {"role": "chair",   "center": (0.16, 0.74), "size_scale": 0.90, "name": "Left Flank Chair"}
            ]
        elif layout_idx == 1:
            # Shifted 90 deg / L-Shape Corner Arrangement
            target_slots = [
                {"role": "seating", "center": (0.26, 0.68), "size_scale": 1.0, "name": "West Lateral Wall Sofa"},
                {"role": "surface", "center": (0.35, 0.83), "size_scale": 0.92, "name": "Corner Conversational Table"},
                {"role": "chair",   "center": (0.76, 0.68), "size_scale": 0.95, "name": "East Corner Reading Chair"},
                {"role": "lighting","center": (0.88, 0.52), "size_scale": 0.92, "name": "East Window Arc Lamp"},
                {"role": "decor",   "center": (0.58, 0.48), "size_scale": 0.88, "name": "Window Greenery Sanctuary"}
            ]
        elif layout_idx == 2:
            # Floating Central Island
            target_slots = [
                {"role": "seating", "center": (0.50, 0.58), "size_scale": 0.98, "name": "Floating Central Sofa"},
                {"role": "surface", "center": (0.50, 0.76), "size_scale": 0.90, "name": "Floating Center Table"},
                {"role": "chair",   "center": (0.22, 0.64), "size_scale": 0.88, "name": "Left Swivel Lounge"},
                {"role": "chair",   "center": (0.78, 0.64), "size_scale": 0.88, "name": "Right Swivel Lounge"},
                {"role": "lighting","center": (0.86, 0.54), "size_scale": 0.88, "name": "Perimeter Corner Lamp"}
            ]
        else:
            # Dual-Zone Multi-Purpose
            target_slots = [
                {"role": "seating", "center": (0.30, 0.66), "size_scale": 0.95, "name": "Zone A Media Sofa"},
                {"role": "surface", "center": (0.30, 0.82), "size_scale": 0.88, "name": "Zone A Coffee Table"},
                {"role": "surface", "center": (0.76, 0.62), "size_scale": 0.90, "name": "Zone B Study Desk"},
                {"role": "chair",   "center": (0.76, 0.78), "size_scale": 0.88, "name": "Zone B Ergonomic Chair"},
                {"role": "decor",   "center": (0.54, 0.52), "size_scale": 0.86, "name": "Zone Divider Planter"}
            ]

        # 4. Canvas for compositing
        result_canvas = clean_bg.copy().convert("RGBA")
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)

        # Categorize extracted sprites
        seating_sprites = [s for s in sprites if s["category"] in ["Seating", "Sleeping"]]
        surface_sprites = [s for s in sprites if s["category"] == "Surface"]
        decor_sprites = [s for s in sprites if s["category"] in ["Decor", "Lighting", "Storage", "General"]]

        # Sort slots back-to-front (smaller Y first so items in front overlap items in back)
        sorted_slots = sorted(target_slots, key=lambda s: s["center"][1])

        for slot in sorted_slots:
            slot_role = slot["role"]
            target_cx = int(slot["center"][0] * w)
            target_cy = int(slot["center"][1] * h)

            sprite_to_use = None
            if slot_role == "seating" and seating_sprites:
                sprite_to_use = seating_sprites.pop(0)
            elif slot_role == "surface" and surface_sprites:
                sprite_to_use = surface_sprites.pop(0)
            elif decor_sprites:
                sprite_to_use = decor_sprites.pop(0)
            elif sprites:
                sprite_to_use = sprites.pop(0)

            # If a detected sprite exists, reposition it!
            if sprite_to_use is not None:
                spr_img = sprite_to_use["image"]
                orig_w, orig_h = spr_img.size
                scale = slot["size_scale"]
                new_w = max(40, int(orig_w * scale))
                new_h = max(30, int(orig_h * scale))
                resized_spr = spr_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Top-left anchor for target center
                paste_x = max(0, min(w - new_w, target_cx - new_w // 2))
                paste_y = max(0, min(h - new_h, target_cy - new_h // 2))

                # Render ground contact drop shadow
                sh_w = int(new_w * 0.90)
                sh_h = max(10, int(new_h * 0.22))
                sh_x1 = paste_x + (new_w - sh_w) // 2
                sh_y1 = paste_y + new_h - int(sh_h * 0.70)
                sh_x2 = sh_x1 + sh_w
                sh_y2 = sh_y1 + sh_h
                shadow_draw.ellipse([sh_x1, sh_y1, sh_x2, sh_y2], fill=(15, 18, 25, 120))

                # Composite sprite
                result_canvas.alpha_composite(resized_spr, (paste_x, paste_y))
            else:
                # Render clean architectural representation for newly introduced slot
                box_w = int(w * (0.42 if slot_role == "seating" else 0.24 if slot_role == "surface" else 0.14))
                box_h = int(h * (0.20 if slot_role == "seating" else 0.12 if slot_role == "surface" else 0.16))
                paste_x = max(0, min(w - box_w, target_cx - box_w // 2))
                paste_y = max(0, min(h - box_h, target_cy - box_h // 2))

                # Draw floor contact shadow
                shadow_draw.ellipse(
                    [paste_x + 5, paste_y + box_h - 15, paste_x + box_w - 5, paste_y + box_h + 10],
                    fill=(15, 18, 25, 100)
                )

                # Render tailored architectural element patch matching style
                patch_img = self._create_architectural_furniture_patch(
                    size=(box_w, box_h),
                    role=slot_role,
                    style=style,
                    user_color=user_color_tone
                )
                result_canvas.alpha_composite(patch_img, (paste_x, paste_y))

        # Blur contact shadow layer and composite beneath objects
        shadow_blurred = shadow_layer.filter(ImageFilter.GaussianBlur(radius=8))
        final_img = Image.alpha_composite(clean_bg.convert("RGBA"), shadow_blurred)
        final_img = Image.alpha_composite(final_img, result_canvas)

        # Subtle aesthetic polish
        out_rgb = final_img.convert("RGB")
        out_rgb = ImageEnhance.Contrast(out_rgb).enhance(1.08)
        out_rgb = ImageEnhance.Sharpness(out_rgb).enhance(1.15)

        return out_rgb

    def _create_architectural_furniture_patch(
        self,
        size: Tuple[int, int],
        role: str,
        style: str,
        user_color: Optional[str] = None
    ) -> Image.Image:
        """
        Creates a high-end tailored furniture sprite patch matching the selected interior style.
        """
        bw, bh = size
        patch = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)

        # Determine primary color based on style
        st_clean = style.lower()
        if "indust" in st_clean:
            base_col = (140, 85, 45, 235)      # Cognac leather
            accent_col = (35, 38, 45, 255)     # Blackened steel
        elif "japan" in st_clean:
            base_col = (215, 195, 170, 235)    # Oatmeal linen / Oak
            accent_col = (165, 135, 105, 255)  # Cedar trim
        elif "scandi" in st_clean:
            base_col = (235, 230, 220, 240)    # Boucle wool
            accent_col = (200, 175, 140, 255)  # Blonde ash
        elif "boho" in st_clean:
            base_col = (195, 105, 55, 235)     # Terracotta
            accent_col = (160, 130, 70, 255)   # Rattan
        elif "bio" in st_clean:
            base_col = (75, 115, 75, 235)      # Sage green
            accent_col = (130, 95, 60, 255)    # Teak wood
        else:
            base_col = (85, 95, 110, 235)      # Slate grey
            accent_col = (30, 35, 45, 255)     # Matte black

        # User color accent if present
        if user_color and user_color.startswith("#") and len(user_color) == 7:
            try:
                hex_c = user_color.lstrip("#")
                r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
                base_col = (int((base_col[0] + r) / 2), int((base_col[1] + g) / 2), int((base_col[2] + b) / 2), 240)
            except Exception:
                pass

        if role == "seating":
            # 3D Pillow / Sectional Silhouette
            draw.rounded_rectangle([4, int(bh * 0.15), bw - 4, bh - 6], radius=14, fill=base_col)
            draw.rounded_rectangle([10, int(bh * 0.05), bw - 10, int(bh * 0.45)], radius=10, fill=accent_col)
            # Cushions
            mid_x = bw // 2
            draw.line([(mid_x, int(bh * 0.18)), (mid_x, bh - 8)], fill=(0, 0, 0, 80), width=2)
        elif role == "surface":
            # Wood / Stone Table Top with Beveled Edge
            draw.rounded_rectangle([6, int(bh * 0.2), bw - 6, bh - 10], radius=8, fill=base_col)
            draw.line([(12, bh - 10), (12, bh - 2)], fill=accent_col, width=4)
            draw.line([(bw - 12, bh - 10), (bw - 12, bh - 2)], fill=accent_col, width=4)
        elif role == "chair":
            # Lounge Armchair
            draw.rounded_rectangle([4, 4, bw - 4, bh - 4], radius=12, fill=base_col)
            draw.rounded_rectangle([10, 8, bw - 10, int(bh * 0.45)], radius=8, fill=accent_col)
        else:
            # Decor / Planter / Lighting
            draw.ellipse([int(bw * 0.2), int(bh * 0.2), int(bw * 0.8), int(bh * 0.8)], fill=base_col)
            draw.ellipse([int(bw * 0.3), int(bh * 0.08), int(bw * 0.7), int(bh * 0.45)], fill=accent_col)

        # Soft edge blur for realism
        patch = patch.filter(ImageFilter.GaussianBlur(radius=1.2))
        return patch


reposition_engine = SpatialRepositionEngine()
