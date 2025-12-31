from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    total_reps = Column(Integer)
    fps = Column(Integer)
    
class RepMetric(Base):
    __tablename__ = "rep_metrics"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    rep_number = Column(Integer)
    knee_angle = Column(Float)
    bar_path_dev = Column(Float)
    quality = Column(String)