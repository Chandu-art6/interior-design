import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Boolean,
)
from sqlalchemy.orm import relationship
from database.session import Base


class User(Base):
    """
    User Model
    Stores user profile, authentication/session tokens, and project references.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=True)
    session_token = Column(String(128), unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # Relationships
    preferences = relationship("Preference", back_populates="user", cascade="all, delete-orphan")
    room_analyses = relationship("RoomAnalysis", back_populates="user", cascade="all, delete-orphan")
    generated_designs = relationship("GeneratedDesign", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Preference(Base):
    """
    Preference Model
    Stores budget constraints, aesthetic style preferences, and multi-user weighting.
    """
    __tablename__ = "preferences"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    budget_tier = Column(String(32), default="mid-range")  # "budget", "mid-range", "luxury"
    budget_amount = Column(Float, nullable=True)
    
    # JSON list of favorite styles (e.g. ["Modern Minimalist", "Japandi", "Scandinavian"])
    style_preferences = Column(JSON, default=list)
    
    # JSON list of preferred color palettes (e.g. ["#F5F5DC", "#2F4F4F", "#D2B48C"])
    color_preferences = Column(JSON, default=list)
    
    # Multi-user preference weights for shared spaces (e.g. {"user_1": 0.6, "user_2": 0.4})
    multi_user_weighting = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<Preference(id={self.id}, user_id={self.user_id}, budget_tier='{self.budget_tier}')>"


class RoomAnalysis(Base):
    """
    RoomAnalysis Model
    Stores computer vision & YOLOv8 detection results, room geometry, and spatial dimensions.
    """
    __tablename__ = "room_analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_uid = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    image_path = Column(String(256), nullable=False)
    room_type = Column(String(64), default="Living Room")
    
    # YOLOv8 detected items (List of dicts: [{"label": "sofa", "confidence": 0.92, "bbox": [0.1, 0.2, 0.5, 0.7]}])
    detected_objects = Column(JSON, default=list)
    
    # Estimated room dimensions (e.g. {"estimated_width_m": 4.5, "estimated_length_m": 5.2, "area_sqm": 23.4})
    spatial_dimensions = Column(JSON, default=dict)
    
    lighting_condition = Column(String(64), default="Natural Daylight")
    clutter_score = Column(Float, default=0.0)  # Clutter index 0.0 to 100.0
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="room_analyses")
    generated_designs = relationship("GeneratedDesign", back_populates="room_analysis", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<RoomAnalysis(id={self.id}, uid='{self.analysis_uid}', room_type='{self.room_type}')>"


class GeneratedDesign(Base):
    """
    GeneratedDesign Model
    Stores generated interior renderings, ControlNet parameters, comfort metrics, and explainable AI logs.
    """
    __tablename__ = "generated_designs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    design_uid = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    room_analysis_id = Column(Integer, ForeignKey("room_analyses.id"), nullable=True, index=True)
    
    style_applied = Column(String(64), nullable=False)
    original_image_path = Column(String(256), nullable=False)
    generated_image_path = Column(String(256), nullable=False)
    prompt_used = Column(Text, nullable=True)
    
    # Comfort & Ergonomic Optimization Scores (0.0 - 100.0)
    circulation_score = Column(Float, nullable=True)
    lighting_score = Column(Float, nullable=True)
    ergonomics_score = Column(Float, nullable=True)
    clutter_index = Column(Float, nullable=True)
    overall_comfort_score = Column(Float, nullable=True)
    
    # Explainable AI Reasoning Logs (List of justifications, architectural rules applied)
    ai_reasoning_log = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="generated_designs")
    room_analysis = relationship("RoomAnalysis", back_populates="generated_designs")

    def __repr__(self):
        return f"<GeneratedDesign(id={self.id}, uid='{self.design_uid}', style='{self.style_applied}')>"
