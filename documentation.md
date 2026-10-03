The Technical Solution: Precision Inpainting
The problem shown is that the entire scene is being modified. To correct this, your technical stack must use explicit segmentation masks and depth guidance to define exactly what to keep, remove, and replace.

You need to switch your generation pipeline to Semantic Inpainting (modifying only targeted areas) rather than Image-to-Image (modifying the whole scene).

Revised Technical Workflow & Components
Component	Technical Implementation
Masking Model (SAM)	Use Meta's Segment Anything Model (SAM) on the backend. Your system must automatically segment specific furniture pieces (e.g., the existing artwork, the existing plant). These segments are the areas we want to replace.
ControlNet (Depth & Segment)	The foundation must remain locked. You use ControlNet trained on depth maps (to preserve perspective) and semantic segmentation (to define objects). The background elements (windows, walls, TV) must have near-perfect masks to prevent "style drift."
Generative Prompting	Do not just prompt "Japandi style." Your prompt must use the spatial information. Example: [background unchanged], [TV unchanged], remove existing artwork, replace with [minimalist textured plaster art], replace plant with [asymmetrical ikebana ceramic piece].
Step-by-Step Implementation of the Correct Workflow
To fix the specific output in your image, your backend pipeline must perform these three steps:

1. Analyze and Segment (The "AI-Vision" Phase)
Instead of applying a global style, the system first identifies the geometry it must preserve and the specific items to replace based on a Japandi style.

Required Masks (Generated Automatically):

Keep: Walls, Windows, Recessed Lights, Area Rug.

Modify (Targeted): Artwork, Plant, Center Vases.

2. Generate the Replacement and Guide the Fill (The "ControlNet" Phase)
The system passes these masks to the generation model along with a ControlNet depth map.

3. Structured Output (The Final Result)
This final image is generated only within the non-masked areas. The scene architecture is untouched, but the specific items are swapped for the new Japandi objects as specified.

Key Takeaway for Your Developers
Tell your backend engineers that the current setup is a basic, global Img2Img / Style Transfer pipeline. To fix it, you need a Masked-Inpainting and ControlNet (Depth/Segment) pipeline. This is the only way to swap decor while keeping the room identical.

If you are using a managed API (like Replicate or Fal.ai), check that you are using an endpoint specifically for ControlNet inpainting with user-provided masks.