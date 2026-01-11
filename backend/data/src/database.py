from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os

#creates a file called squat_optimizer.db in the current directory
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./squat_optimizer.db")

#need check_same_thread=False to work with FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()