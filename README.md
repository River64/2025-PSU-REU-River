# 2025-PSU-REU-River

## Context
The code in the repository this one was forked from was used in the [publication](https://pdxscholar.library.pdx.edu/altreu_projects/17/) Capturing Non-motorized Counts at Intersections Using Ultralytics YOLOv8 Image and Video Tagging by Alicia Hopper, Tammy Lee, and Sirisha Kothuri. It was written by Alicia Hopper, during a 2024 NSF REU program at Portland State University (PSU). This repository contains edits by River Johnson, during a 2025 REU program at the same university.
## Purpose
The code in this project is used to train and validate a YOLO model on data from the EuroCity Persons dataset. Below is a description of each file in the repository, the filesystem structure used to set up the dataset, instructions for setting up the dataset in that structure, and how to run the model.

As a general note, it is expected that you're running code in the repository directory from within that directory (since I wanted to use regular relative file paths instead of messing around with `os`). I may change this in the future, but currently it impacts `dataconverter.py` and `ECP.yaml`.
## Code
ECP.yaml
countObjects.py
dataconverter.py
test_cuda_available.py
train.py
## File structure
Project

├ 2025-PSU-REU-River (this code) (can have any name)

├	datasets (any datasets intended to be used for training or validation)

    └ ECP (eurocity persons database) -- can be obtained from https://eurocity-dataset.tudelft.nl/eval/overview/statistics

        ├ images

            ├ train (training images)

            └ val (validation images)
            
        └ labels

            ├ train (training labels - one corresponding to each image)

            └ val (val labels - one corresponding to each image)

        └ old_labels (folder to put original labels in ECP format before conversion into YOLO format)
## Dataset formatting
### Labels
Use `dataconverter.py`
### Images (flat folder structure)
## Running remotely
Use a [virtual environment](https://docs.python.org/3/library/venv.html) if necessary to install the ultralytics package. This will most likely be necessary on a remote server, because you won't have permission to edit the server environment.
1. `python -m venv` to create a new virtual environment
2. `source .venv/bin/activate` to activate
3. `pip install ultralytics` to install the ultralytics package (after activating the virtual environment)
4. Run this code while the virtual environment is activated
5. `deactivate` to deactivate

Use [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) if running on a remote machine, so that training (and/or other time-consuming tasks) doesn't get interrupted by your computer losing power or connection to the server.
1. `tmux new` to create a new session
2. Start running some time-consuming command (for example, `python3 train.py`)
3. `ctrl+b then press d` to detach from the session
4. `tmux attach` to re-attach and check in later
