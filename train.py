"""
This program is used to train a YOLO model on ECP data

Written by Allie Hopper, 2024
Edited by River Johnson, 2025
"""

from ultralytics import YOLO
import torch


# ----------- DEBUGGING ----------- #

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

print(torch.__version__)
# --------------------------------- #

# I never want to accidentally run this using the CPU. Disable this check to run on a cpu
if device == "cpu":
    exit()

# Load a pretrained YOLO model. For initial training, use "yolov8s.pt"
# model = YOLO("runs/detect/train6/weights/best.pt")
model = YOLO("yolov10m.pt")

# Train the model using the 'ECP.yaml' dataset for 300 epochs - set resume=True to resume interrupted training
# Original: results = model.train(data="2024-PSU-REU/ECP.yaml", epochs=100, imgsz=640, workers=16, resume=True)
# Current version:
results = model.train(data="ECP.yaml", epochs=300, imgsz=640, workers=16, device=0, cache=True, patience=50)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image using the model
results = model.predict("image.png")

# Export the model to ONNX format
success = model.export(format="onnx")
