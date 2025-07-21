from ultralytics import YOLO

# Based on code from https://docs.ultralytics.com/guides/hyperparameter-tuning/#custom-search-space-example 

# Initialize the YOLO model
model = YOLO("yolov10m.pt")

# Define hyperparameter search space
# Use space=search_space in the tune function if default
# search space takes too long
search_space = {
        "lr0": (1e-5, 1e-1),
}

# Tune hyperparameters on ECP data for 30 epochs
model.tune(
        data="ECP.yaml",
        epochs=20,
        iterations=50,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False,
        # Normal model training parameters:
        imgsz=640,
        device=0,
        cache=True,
)

