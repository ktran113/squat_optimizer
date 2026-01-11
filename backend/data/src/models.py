from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column (String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=True)
    video_path =Column(String, nullable=False)
    fps = Column(Integer, default=30)
    total_reps = Column(Integer, nullable=False)
    #metrics
    avg_depth=Column(Float)
    min_knee_angle =Column(Float)
    tempo = Column(Float)
    alignment = Column(Float)
    bar_dev = Column(Float)
    ai_feedback = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    reps = relationship("RepMetric", back_populates="session", cascade="all, delete-orphan")


class RepMetric(Base):
    __tablename__ = 'rep_metrics'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'))
    rep_number = Column(Integer, nullable=False)
    
    bottom_frame = Column(Integer)
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    
    knee_angle = Column(Float)
    depth_value = Column(Float)
    depth_quality = Column(String)  # 'below', 'parallel', 'partial'
    bar_path_deviation = Column(Float)
    tempo = Column(Float)
    hip_heel_aligned = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="reps")