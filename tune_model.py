from ultralytics import YOLO

# Based on code from https://docs.ultralytics.com/guides/hyperparameter-tuning/#custom-search-space-example 

# Initialize the YOLO model
model = YOLO("yolov10s.pt")

# Define hyperparameter search space
# Use space=search_space in the tune function if default
# search space takes too long
search_space = {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98),
        "warmup_epochs": (0.0, 5.0),
        "warmup_momentum": (0.0, 0.95),
        "hsv_h": (0.0, 0.1),
        "scale": (0.0, 0.9),
}

# Tune hyperparameters on ECP data for 30 epochs
model.tune(
        data="ECP.yaml",
        epochs=30,
        iterations=50,
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

