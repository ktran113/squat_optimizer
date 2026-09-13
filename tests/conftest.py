import os
import tempfile

#auth.py refuses to import without a secret and database.py binds its engine at
#import time, so these must be set before any app module is loaded. Setting them
#here also keeps load_dotenv() from pulling in values from a real .env
os.environ["JWT_SECRET_KEY"] = "test-secret"
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"
os.environ["OPENAI_API_KEY"] = ""   #feedback.py returns a fallback message instead of calling OpenAI

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from database import engine
from models import Base

REP_PERIOD = 90         #frames per rep, 3s at 30fps
STANDING_HIP_Y = 200
KNEE = (300, 400)
ANKLE = (300, 600)


@pytest.fixture
def client():
    #fresh tables per test so tests do not depend on each other
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def make_squat():
    """
    Builds side-view pose keypoints for a lifter squatting with a fixed knee and
    ankle while the hip moves down and back up. Rep bottoms land on frames
    45, 135, 225, ... With the hip 100px behind the knee, a bottom hip y of 420
    is ~79 degrees (below), 391 is ~95 (parallel), and 340 is ~121 (partial).
    tremor adds a fast wobble to mimic camera shake.
    """
    def build(n_frames=300, bottom_y=420, tremor=0.0):
        t = np.arange(n_frames)
        amplitude = (bottom_y - STANDING_HIP_Y) / 2
        hip_y = (STANDING_HIP_Y + amplitude * (1 - np.cos(2 * np.pi * t / REP_PERIOD))
                 + tremor * np.sin(2 * np.pi * t / 6))

        xy = np.zeros((n_frames, 17, 2), dtype=np.float32)
        for hip, knee, ank in [(11, 13, 15), (12, 14, 16)]:
            xy[:, hip, 0] = 200
            xy[:, hip, 1] = hip_y
            xy[:, knee] = KNEE
            xy[:, ank] = ANKLE
        conf = np.full((n_frames, 17), 0.9, dtype=np.float32)
        return xy, conf

    return build
