import os
import urllib.request
from ultralytics import YOLO
 
URLS = [
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n-seg.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n-pose.pt",

    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26s.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26s-seg.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26s-pose.pt",

    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26m.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26m-seg.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26m-pose.pt",

    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26l.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26l-seg.pt",
    "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26l-pose.pt",
]

models = [
    "yolo26n.pt",
    "yolo26n-seg.pt",
    "yolo26n-pose.pt",

    "yolo26s.pt",
    "yolo26s-seg.pt",
    "yolo26s-pose.pt",

    "yolo26m.pt",
    "yolo26m-seg.pt",
    "yolo26m-pose.pt",

    "yolo26l.pt",
    "yolo26l-seg.pt",
    "yolo26l-pose.pt",

    "yolo26x.pt",
    "yolo26x-seg.pt",
    "yolo26x-pose.pt",

    "yolo26n-depth.pt",
    "yolo26s-depth.pt",
    "yolo26m-depth.pt",
    "yolo26l-depth.pt",
    "yolo26x-depth.pt",
]

def process_file(filename):
    print(f"Processing {filename}")

    base, _ = os.path.splitext(filename)
    output = base + ".onnx"

    if os.path.exists(output):
        print(f"Already processed: {output}")
        return

    model = YOLO(filename) 
    model.export(
     format="onnx",
     opset=12,        # OpenCV 5 DNN needs opset >= 12
     imgsz=640,       # fixed square input
     dynamic=False,   # static shapes avoid a known DNN parsing issue
     simplify=True,   # fold constants for a cleaner graph
     nms=False,       # keep the raw head; YOLO26 is already NMS-free
    )

def process_file_q8(filename):
    print(f"Processing {filename}")

    base, _ = os.path.splitext(filename)
    output = base + "_int8.onnx"

    if os.path.exists(output):
        print(f"Already processed: {output}")
        return

    model = YOLO(filename) 
    model.export(
     format="onnx",
     opset=12,        # OpenCV 5 DNN needs opset >= 12
     imgsz=640,       # fixed square input
     dynamic=False,   # static shapes avoid a known DNN parsing issue
     simplify=True,   # fold constants for a cleaner graph
     nms=False,       # keep the raw head; YOLO26 is already NMS-free
     quantize=8
    )


for url in models:
    filename = os.path.basename(url)

    process_file(filename)
    # process_file_q8(filename)