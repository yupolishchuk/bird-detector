import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA = "data.yaml"
MODEL = "yolo26n.pt"
IMGSZ = 1280
BATCH = 8
SEED = 42
PATIENCE = 100
LEARNING_RATE = 0.0005

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

model = YOLO(MODEL)
model.train(
    data=DATA,
    epochs=args.epochs,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=SEED,
    patience=PATIENCE,
    optimizer="AdamW",
    lr0=LEARNING_RATE,
    project="runs",
    name="bird",
)

Path("weights").mkdir(exist_ok=True)
shutil.copy(model.trainer.best, "weights/best.pt")

