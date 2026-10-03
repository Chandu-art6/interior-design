import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database.connection import init_db
from app.api.routes import router as api_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API for Generative AI-Based Interior Space Redesign with Comfort Optimization.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit / Frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for uploaded and generated images
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Mount API routes
app.include_router(api_router)


@app.on_event("startup")
async def on_startup():
    """Initialize database tables on application start."""
    init_db()


@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": settings.DEVICE
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
