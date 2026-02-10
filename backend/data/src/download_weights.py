"""
Downloads the barbell dataset from Roboflow and trains a YOLOv8n model locally.
Run once: python download_weights.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "barbell_dataset")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "barbell", "weights", "best.pt")

def download_and_train():
    if os.path.exists(WEIGHTS_PATH):
        print(f"weights already at {WEIGHTS_PATH}")
        return

    data_yaml = os.path.join(DATASET_DIR, "data.yaml")
    if not os.path.exists(data_yaml):
        import roboflow
        rf = roboflow.Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(os.getenv("ROBOFLOW_PROJECT"))
        version = project.version(int(os.getenv("ROBOFLOW_VERSION")))
        version.download("yolov8", location=DATASET_DIR)
    else:
        print(f"Dataset already exists")

    # Train YOLOv8n
    from ultralytics import YOLO
    print("training")
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        project=os.path.join(os.path.dirname(__file__), "..", "models"),
        name="barbell",
        exist_ok=True
    )
    print(f"done weights saved to {WEIGHTS_PATH}")

if __name__ == "__main__":
    download_and_train()
