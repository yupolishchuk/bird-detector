import argparse

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--weights", default="weights/best.pt") # для того щоб експерементувати з вагами
parser.add_argument("--threshold", type=float, default=0.25) # threshold нижче якого рамка не накладається 
args = parser.parse_args()

IMGSZ = 1280

model = YOLO(args.weights)

input = cv2.VideoCapture(args.input)
if not input.isOpened():
    raise SystemExit(f"Не вдалося відкрити {args.input}")

fps = input.get(cv2.CAP_PROP_FPS)
size = (int(input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

while True:
    ok, frame = input.read()
    if not ok:
        break
    result = model(frame, conf=args.threshold, imgsz=IMGSZ, verbose=False)[0]
    output.write(result.plot())

input.release()
output.release()
