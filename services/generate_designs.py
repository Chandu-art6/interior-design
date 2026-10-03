import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os


# ============================================================
# 1. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# ============================================================
# 2. LOAD MODEL
# ============================================================

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

# Optional memory optimization
if device == "cuda":
    pipe.enable_attention_slicing()


# ============================================================
# 3. GENERATE ONE DESIGN
# ============================================================

def generate_design(
    input_image,
    style,
    furniture_layout,
    color_scheme,
    budget,
    output_file
):

    prompt = f"""
    Professional interior redesign of the EXACT SAME ROOM shown
    in the input photograph.

    IMPORTANT:
    Preserve the physical architecture of the room:
    walls, ceiling, floor structure, doors, windows,
    room proportions and architectural openings.

    Do NOT redesign the architecture.

    Completely redesign the movable interior furniture and decoration.

    Interior style:
    {style}

    Furniture arrangement:
    {furniture_layout}

    Color scheme:
    {color_scheme}

    Budget level:
    {budget}

    Requirements:

    - substantially different furniture arrangement
    - different sofa design
    - different chairs
    - different coffee table
    - different TV/media unit
    - different rug
    - different curtains
    - different lighting
    - different wall decoration
    - realistic furniture sizes
    - realistic spacing
    - comfortable walking paths
    - doors and windows must remain accessible
    - practical interior design
    - photorealistic
    - realistic materials
    - realistic lighting
    - high quality architectural visualization

    Do NOT simply recolor the existing furniture.

    Do NOT copy the original furniture arrangement.

    The result must look like a genuinely redesigned interior.
    """

    negative_prompt = """
    duplicate furniture,
    same furniture arrangement,
    copied furniture,
    minor changes,
    recolored original room,
    distorted furniture,
    floating furniture,
    unrealistic furniture,
    blocked door,
    blocked window,
    distorted walls,
    changed architecture,
    extra doors,
    extra windows,
    deformed objects,
    blurry,
    low quality,
    cartoon,
    unrealistic proportions
    """

    # Convert image
    image = input_image.convert("RGB")

    # Resize
    image = image.resize((768, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=0.70,
        guidance_scale=8.0,
        num_inference_steps=40
    ).images[0]

    result.save(output_file)

    print("Generated:", output_file)

    return result


# ============================================================
# 4. GENERATE MULTIPLE DIFFERENT DESIGNS
# ============================================================

def generate_multiple_designs(input_path):

    os.makedirs("generated_designs", exist_ok=True)

    original = Image.open(input_path)

    # --------------------------------------------------------
    # OPTION 1
    # --------------------------------------------------------

    option1 = generate_design(
        input_image=original,

        style="Modern Minimalist",

        furniture_layout="""
        Place a modern three-seater sofa against the back wall.
        Put a rectangular coffee table in the center.
        Place one accent chair on the left side
        and another accent chair on the right side.
        Add a low modern TV console on the opposite wall.
        Keep the center walking area open.
        """,

        color_scheme="""
        white, beige, light grey and natural light wood
        """,

        budget="medium",

        output_file="generated_designs/option1_modern.jpg"
    )


    # --------------------------------------------------------
    # OPTION 2
    # --------------------------------------------------------

    option2 = generate_design(
        input_image=original,

        style="Luxury Contemporary",

        furniture_layout="""
        Place a large premium sofa along the left wall.
        Place two designer lounge chairs diagonally
        on the opposite side.
        Put a large round coffee table slightly
        toward the center-right.
        Add a premium wall-mounted TV and elegant media console.
        Add decorative side tables and statement lighting.
        """,

        color_scheme="""
        cream, charcoal, dark brown, warm gold
        and premium wood
        """,

        budget="high",

        output_file="generated_designs/option2_luxury.jpg"
    )


    # --------------------------------------------------------
    # OPTION 3
    # --------------------------------------------------------

    option3 = generate_design(
        input_image=original,

        style="Scandinavian Japandi",

        furniture_layout="""
        Place an L-shaped sofa along the back wall
        and left wall.
        Put a small round wooden coffee table in the center.
        Place one compact wooden lounge chair
        near the right corner.
        Use a slim low-profile TV cabinet.
        Keep the room spacious and uncluttered.
        """,

        color_scheme="""
        cream, beige, soft brown,
        natural wood and muted green
        """,

        budget="medium",

        output_file="generated_designs/option3_japandi.jpg"
    )


    print("\n===================================")
    print("3 INTERIOR DESIGNS GENERATED")
    print("===================================")

    print("Option 1:", option1)
    print("Option 2:", option2)
    print("Option 3:", option3)


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":

    input_image_path = "input_room.jpg"

    generate_multiple_designs(input_image_path)