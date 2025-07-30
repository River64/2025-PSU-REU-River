# 2025-PSU-REU-River

## Context
The code in the repository this one was forked from was used in the [publication](https://pdxscholar.library.pdx.edu/altreu_projects/17/) Capturing Non-motorized Counts at Intersections Using Ultralytics YOLOv8 Image and Video Tagging by Alicia Hopper, Tammy Lee, and Sirisha Kothuri. It was written by Alicia Hopper, during a 2024 NSF REU program at Portland State University (PSU). This repository contains edits by River Johnson, during a 2025 REU program at PSU.
## Purpose
The code in this project is used to train and validate a YOLO model on data from the EuroCity Persons dataset. Below is a [description](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#code-description) of each file in the repository, the [filesystem structure](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#file-structure) used to set up the dataset, [instructions](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#dataset-formatting) for setting up the dataset in that structure, and [tips](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#running-remotely) for training the model.

As a general note, it is expected that you're running code in the repository directory from within that directory (since I wanted to use regular relative file paths instead of messing around with the python `os` module). I may change this in the future, but currently it impacts `dataconverter.py` and `ECP.yaml`.
## Code description
`dataconverter.py` converts labels from ECP format to a format the YOLO model uses, as well as excluding bounding boxes for labels that should not be included. Currently it only excludes far-away labels. It also combines rider bounding boxes with the corresponding rider type.

`ECP.yaml` describes the ECP dataset: the file paths that the dataset is contained in and each object class to identify. This is a file that is required for running a YOLO model. It is used in `train.py`.

`train.py` trains a YOLO model on ECP data for a specified number of epochs. See comments in the file itself for more detailed information and examples.

`test_cuda_available.py` is a short script to use to check whether CUDA is available in the environment. This check is also included in `train.py` (and that script will quit if CUDA is using the CPU), but it's also convenient to have and run by itself. "Using cuda device" means that the GPU will be used in training the model, which is ideal.

`countObjects.py` uses a trained model to count objects within a video.
## File structure
This is mainly based on the required [folder structure](https://docs.ultralytics.com/datasets/classify/) for YOLO model training.
```
Project (parent folder, can have any name)

├ 2025-PSU-REU-River (this code) (can have any name)

├ datasets (any datasets intended to be used for training or validation)

    └ ECP (eurocity persons database) -- can be obtained from https://eurocity-dataset.tudelft.nl

        ├ images

            ├ train (training images)

            └ val (validation images)
            
        └ labels

            ├ train (training labels - one corresponding to each image)

            └ val (val labels - one corresponding to each image)

        └ old_labels (folder to put original labels in ECP format before conversion into YOLO format)
```
## Dataset formatting
### Images
There are two options for the image paths:
1. Moving the images out of their city folders
```
mv */* . (run from the train and val directories to move the files out of the city folders)
rmdir `ls -ld` (to remove the resulting empty city folders)
```
2. Editing the paths used in ECP.yaml: from `images/train` to `images/train/*`, and `images/val` to `images/val/*`. I have not tested this personally, but it should work, by using wildcards to ignore the city folder names. If it doesn't work (for example, if it messes up the paths to the labels), just go with method 1.
### Labels
Here is the structure of the old_labels folder, which matches the format of the ECP dataset:
```
old_labels

├ train

    └ 31 city folders

        └ city_labelnumber.json (~500-1000 for each city)

└ val

    └ 31 city folders

        └ city_labelnumber.json (~100-200 for each city)
```
Run `dataconverter.py` to convert the training and validation label files. Files in old_labels will not be destroyed or modified. New label files will be created in the labels folder.
## Running remotely
### Virtual environment
Use a [virtual environment](https://docs.python.org/3/library/venv.html) if necessary to install the ultralytics package. This will likely be necessary on a remote server, because you won't have permission to edit the server environment.
1. `python -m venv` to create a new virtual environment
2. `source .venv/bin/activate` to activate
3. `pip install ultralytics` to install the ultralytics package (after activating the virtual environment)
4. Run any code that imports the ultralytics package while the virtual environment is activated
5. `deactivate` to deactivate (it can be activated again at any point)

### tmux
Use [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) if running on a remote machine, so that training (and/or other time-consuming tasks) doesn't get interrupted by your computer losing power or connection to the server.
1. `tmux new` to create a new session
2. Start running some time-consuming command (for example, `python3 train.py`)
3. `ctrl+b then press d` to detach from the session
4. `tmux attach` to re-attach and check in later

### Overall process
Setup:
1. Create a virtual environment if needed
2. Activate that environment
3. `pip install ultralytics`

Training the model:
1. Create a new tmux session
2. `source .venv/bin/activate` to activate your virtual environment
3. `python3 train.py`
4. `ctrl+b then press d` to detach from the tmux session
5. `tmux attach` to check in later
