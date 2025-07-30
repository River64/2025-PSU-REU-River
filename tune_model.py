from ultralytics import YOLO

"
Written by River Johnson, 2024,
Based on code from https://docs.ultralytics.com/guides/hyperparameter-tuning/#custom-search-space-example 
This code finds the best values of various model hyperparameters
and saves the results in runs/detect/tune.

It is NOT recommended to run this on a regular desktop computer unless you have a month or so
of time on your hands. For example, here are the time calculations for
running this on an NVIDIA RTX A2000 GPU:
7 minutes/epoch (YOLOv10) * 30 epochs * 300 iterations / 1440 minutes/day = 43.75 days
However, this process should be much faster on any multi-GPU setup.
"

# Initialize the YOLO model
model = YOLO("yolov10s.pt")

# This is a custom search space. The values for each hyperparameter are taken from the default search space,
# but only some hyperparameters are included.
# Use space=search_space in the tune function if default search space takes too long to find any useful results
search_space = {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98),
        "warmup_epochs": (0.0, 5.0),
        "warmup_momentum": (0.0, 0.95),
        "hsv_h": (0.0, 0.1),
        "scale": (0.0, 0.9),
}

# Tune hyperparameters on ECP data for 30 epochs each iteration
# 300 iterations is a recommended default-
# Using fewer iterations can lead to not finding useful results
results = model.tune(
        data="ECP.yaml",
        epochs=30,
        iterations=300,
        optimizer="AdamW",
        space=search_space,
        plots=False,
        save=False,
        val=False,
        # Normal model training parameters:
        imgsz=640,
        device=0,
        cache=True,
        workers=16,
)

