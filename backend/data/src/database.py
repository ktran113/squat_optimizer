from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/squat_tracker")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()