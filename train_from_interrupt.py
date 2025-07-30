"""
This program is used to train a YOLO model on ECP data
It is mostly the same as train.py, but with slight edits for
resuming from an interrupted training session.

Written by Allie Hopper, 2024
Edited by River Johnson, 2025
"""

from ultralytics import YOLO
import torch
import sys

if (len(sys.argv) < 2):
    print("Usage: python3 train_from_interrupt.py <folder to resume from>\n
          Example: python3 train_from_interrupt.py train19\n
          These folders are in runs/detect, made when any training run is started")
    exit()
else:
    resume_folder = sys.argv[1]

# ----------- CPU CHECK ----------- #

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

# I never want to accidentally run this using the CPU
# Disable this check to run on a CPU
if device == "cpu":
    exit()
# --------------------------------- #

# Load a pretrained YOLO model. This is the interrupted training run to resume from
# Depending on what run got interrupted, the folder name will be different, so that's a command line argument.
# By default YOLO names the training run folders train#, but there are ways to put in custom names.
# Also, YOLO makes "runs" and its subdirectories in whatever folder you run this script by default.
model = YOLO("runs/detect/" + resume_folder + "/weights/last.pt")

# Train the model using the 'ECP.yaml' dataset for 200 epochs
results = model.train(data="ECP.yaml", epochs=200, imgsz=640, workers=16, device=0, cache=True, resume=True, patience=50)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image using the model
# results = model.predict("image.png")

# Export the model to ONNX format
success = model.export(format="onnx")
