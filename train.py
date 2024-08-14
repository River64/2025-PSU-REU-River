"""
This program is used to train a YOLO model for 100 epochs on one training batch of ECP data

Written by Allie Hopper, 2024
"""

from ultralytics import YOLO

# Load a pretrained YOLO model from the last training batch (if first training batch, use "yolov8n.pt") 

model = YOLO("yolov8n.pt")

# Train the model using the 'ECP.yaml' dataset for 100 epochs
results = model.train(data="2024-PSU-REU/ECP.yaml", epochs=100, imgsz=640, patience=250)

# Evaluate the model's performance on the validation set
# results = model.val()

