"""
Unified Application Launcher
Starts both the FastAPI Backend Server and Streamlit Interactive UI concurrently.

Usage:
    python run_app.py
"""
import sys
import time
import subprocess
import threading

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uvicorn
from config import settings
from database.session import init_db

def run_backend():
    """Runs FastAPI backend using uvicorn server."""
    print(f"[Backend] Initializing Database and launching FastAPI on http://{settings.API_HOST}:{settings.API_PORT}...")
    init_db()
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level="info"
    )

def run_frontend():
    """Runs Streamlit frontend UI."""
    print(f"[Frontend] Launching Streamlit Dashboard on http://localhost:{settings.UI_PORT}...")
    time.sleep(1.5)  # Brief wait for backend startup
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/app.py",
        "--server.port",
        str(settings.UI_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    print("=" * 70)
    print("AI INTERIOR SPACE REDESIGN & COMFORT OPTIMIZER")
    print("=" * 70)
    
    # Start Backend in daemon thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Run Frontend in main thread
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\nShutting down AI Interior Designer services gracefully.")
        sys.exit(0)
