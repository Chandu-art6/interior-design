# 🛋️ AI Interior Space Redesign with Explainable Recommendations & Comfort Optimization

An end-to-end AI-powered interior redesign system that preserves room structural geometry via **ControlNet (Canny / M-LSD / Depth)**, detects furniture using **YOLOv8**, and evaluates **Explainable Ergonomic & Comfort Scores** (Circulation, Natural Lighting, Clutter Index).

---

## 📁 Project Architecture

```
interior-design/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # REST endpoints (Upload, Analyze, Redesign, History)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── design.py              # Pydantic schemas (Detections, Requests, Comfort Metrics)
│   ├── __init__.py
│   └── main.py                    # FastAPI application server entrypoint
├── database/
│   ├── __init__.py
│   ├── connection.py              # SQLite Engine & sessionmaker
│   └── models.py                  # SQLAlchemy models (DesignProject, DetectedObject)
├── services/
│   ├── __init__.py
│   ├── cv_service.py              # OpenCV edge & structural geometry extraction
│   ├── yolo_service.py            # Ultralytics YOLOv8 object & furniture detection
│   ├── controlnet_service.py      # Stable Diffusion + ControlNet diffusion pipelines
│   └── recommender.py             # Comfort metrics scoring & explainability engine
├── ui/
│   ├── components/
│   │   ├── __init__.py
│   │   └── widgets.py             # Custom styled components & metric cards
│   ├── __init__.py
│   └── app.py                     # Streamlit web application frontend
├── static/
│   ├── uploads/                   # Uploaded raw room images
│   └── outputs/                   # Redesigned rendering outputs
├── .env.example                   # Example environment configurations
├── config.py                      # Global settings & model configurations
├── requirements.txt               # Python package dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Backend Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive Swagger API docs available at: `http://127.0.0.1:8000/docs`

### 4. Run Streamlit UI
```bash
streamlit run ui/app.py --server.port 8501
```
Web interface available at: `http://localhost:8501`
