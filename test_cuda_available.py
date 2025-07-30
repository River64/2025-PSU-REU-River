import torch

"
A small script to test if CUDA is available for using the GPU.
This check is run in train.py itself, but having it on its own
is also useful.
"

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
