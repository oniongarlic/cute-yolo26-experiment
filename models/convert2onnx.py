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
]

def process_file(filename):
    print(f"Processing {filename}")
    model = YOLO(filename) 
    model.export(
     format="onnx",
     opset=12,        # OpenCV 5 DNN needs opset >= 12
     imgsz=640,       # fixed square input
     dynamic=False,   # static shapes avoid a known DNN parsing issue
     simplify=True,   # fold constants for a cleaner graph
     nms=False,       # keep the raw head; YOLO26 is already NMS-free
    )


for url in URLS:
    filename = os.path.basename(url)

    if not os.path.exists(filename):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"Already exists: {filename}")

    process_file(filename)