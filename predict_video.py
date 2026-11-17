import argparse

import cv2
from ultralytics import YOLO

IMGSZ = 1280

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--weights", default="weights/best.pt") # для того щоб експериментувати з вагами
parser.add_argument("--threshold", type=float, default=0.4) # threshold нижче якого рамка не накладається
args = parser.parse_args()

model = YOLO(args.weights)

video_in = cv2.VideoCapture(args.input)
if not video_in.isOpened():
    raise SystemExit(f"Не вдалося відкрити {args.input}")

fps = video_in.get(cv2.CAP_PROP_FPS)
size = (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

while True:
    ok, frame = video_in.read()
    if not ok:
        break
    result = model(frame, conf=args.threshold, imgsz=IMGSZ, verbose=False)[0]
    video_out.write(result.plot())

video_in.release()
video_out.release()
