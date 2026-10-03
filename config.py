import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Application Settings and Configuration.
    Handles environment variables, storage paths, AI model checkpoints, and server configs.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    STATIC_DIR: Path = BASE_DIR / "static"
    UPLOAD_DIR: Path = STATIC_DIR / "uploads"
    OUTPUT_DIR: Path = STATIC_DIR / "outputs"

    # Server Settings
    APP_NAME: str = "AI Interior Space Redesign & Comfort Optimizer"
    APP_VERSION: str = "1.0.0"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    UI_PORT: int = 8501
    DEBUG: bool = True

    # Database Configuration
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'database' / 'interior_design.db'}"

    # AI & Compute Device Configuration
    # Options: "cuda", "mps", "cpu", or "auto"
    DEVICE: str = "auto"
    TORCH_DTYPE: str = "float16"  # "float16" for GPU, "float32" for CPU

    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openrouter/free"


    # Object Detection (YOLO / MediaPipe)
    YOLO_MODEL_NAME: str = "yolov8m.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35
    FURNITURE_CLASSES: List[str] = [
        "chair", "couch", "bed", "dining table", "tv", "refrigerator",
        "desk", "lamp", "wardrobe", "bookshelf", "plant", "rug"
    ]

    # Generative AI & ControlNet Architecture
    STABLE_DIFFUSION_MODEL_ID: str = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_CANNY_MODEL_ID: str = "lllyasviel/control_v11p_sd15_canny"
    CONTROLNET_MLSD_MODEL_ID: str = "lllyasviel/control_v11p_sd15_mlsd"
    CONTROLNET_DEPTH_MODEL_ID: str = "lllyasviel/control_v11f1p_sd15_depth"

    # Image Generation Defaults
    IMAGE_WIDTH: int = 512
    IMAGE_HEIGHT: int = 512
    DEFAULT_INFERENCE_STEPS: int = 30
    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_CONTROLNET_SCALE: float = 0.45
    SIMILARITY_THRESHOLD: float = 0.65
    MAX_REGENERATION_ATTEMPTS: int = 2

    # Explainable Recommendation & Comfort Metrics
    DEFAULT_ROOM_TYPES: List[str] = [
        "Living Room", "Bedroom", "Home Office", "Kitchen", "Dining Room", "Studio"
    ]
    DEFAULT_INTERIOR_STYLES: List[str] = [
        "Modern Minimalist", "Scandinavian", "Industrial", "Bohemian",
        "Japandi", "Mid-Century Modern", "Luxury Contemporary", "Biophilic"
    ]
    CIRCULATION_CLEARANCE_MIN_CM: float = 75.0  # Minimum walking clearance in cm
    LIGHTING_COMFORT_OPTIMAL_LUX: float = 300.0


# Initialize global configuration instance
settings = AppSettings()

# Ensure required directories exist at runtime
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.BASE_DIR / "database", exist_ok=True)
