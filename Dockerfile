FROM python:3.13-slim-bookworm

#opencv (pulled in by ultralytics) links against these even when no display is used
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#CPU-only torch first, otherwise ultralytics pulls the multi-GB CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/data/src backend/data/src
COPY backend/data/models/barbell/weights/best.pt backend/data/models/barbell/weights/best.pt
#detect_pose.py loads yolov8s-pose.pt from the working directory
COPY backend/data/models/yolov8s-pose.pt backend/data/src/yolov8s-pose.pt

#keep the SQLite file in its own directory so it can be mounted as a volume
ENV DATABASE_URL=sqlite:////app/data/squat_optimizer.db
RUN mkdir -p /app/data

WORKDIR /app/backend/data/src
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
