# 2025-PSU-REU-River

## Context
The code in the repository this one was forked from was used in the [publication](https://pdxscholar.library.pdx.edu/altreu_projects/17/) Capturing Non-motorized Counts at Intersections Using Ultralytics YOLOv8 Image and Video Tagging by Alicia Hopper, Tammy Lee, and Sirisha Kothuri. It was written by Alicia Hopper, during a 2024 NSF REU program at Portland State University (PSU).

This repository contains significant edits by River Johnson, during a 2025 NSF REU program at Portland State University. The code in this repository was used in Comparing Ultralytics YOLOv8 and YOLOv10 for Multimodal Transportation Counts.

## Purpose
The code in this project is used to train and validate a YOLO model on data from the EuroCity Persons dataset. Below is a [description](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#code-description) of each file in the repository, the [filesystem structure](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#file-structure) used to set up the dataset, [instructions](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#dataset-formatting) for setting up the dataset in that structure, and [tips](https://github.com/River64/2025-PSU-REU-River/tree/main?tab=readme-ov-file#running-the-model) for training the model.

As a general note, it is expected that you're running code in the repository directory from within that directory (since I wanted to use regular relative file paths instead of messing around with the python `os` module). I may change this in the future, but currently it impacts `data_converter.py` and `ECP.yaml`.

## Code description
`ECP.yaml` describes the ECP dataset: the file paths that the dataset is contained in and each object class to identify. This is a file that is required for running a YOLO model. It is used in `train.py`.

`train.py` trains a YOLO model on ECP data for a specified number of epochs. See comments in the file itself for more detailed information and examples.

`train_from_interrupt.py` is very similar to `train.py`, except that it will resume interrupted training given a folder to resume from as a command line argument.

`test_cuda_available.py` is a short script to use to check whether CUDA is available in the environment. This check is also included in `train.py` (and that script will quit if CUDA is using the CPU), but it's also convenient to have and run by itself. "Using cuda device" means that the GPU will be used in training the model, which is ideal.

`tune_model.py` starts a hyperparameter tuning session. This is NOT recommended to run on any regular desktop computer unless you have a lot of time on your hands. See in-file comments for more details.

`count_objects.py` uses a trained model to count objects within a video. This is the least configured and thoroughly tested of all of the programs in this repository.

### Scripts folder
`download_ecp_dataset.sh` runs the suggested wget commands from the ECP dataset website for day training and validation images/labels, using command-line arguments to pass in your ECP username and password. If you don't want to use this script, just download the zip files and place them in a directory named ECP_working under the parent project directory.

`ecp_dataset_setup.sh` unzips the ECP dataset files, runs data_converter.py for the image labels, unpacks the files from the city directories, and deletes the resulting empty city directories.

`data_converter.py` converts labels from ECP format to a format the YOLO model uses, as well as excluding bounding boxes for labels that should not be included: far-away labels, based on level of occlusion, etc. It also combines rider bounding boxes with the corresponding rider type. This is the file to edit to change the dataset configuration.

## File structure
This is based on the required [folder structure](https://docs.ultralytics.com/datasets/classify/) for YOLO model training.
```
Project (parent folder, can have any name)

├ 2025-PSU-REU-River (this code) (can have any name)

    └ scripts (helpful scripts for dataset setup)

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
Run `download_ecp_dataset.sh` to download the ECP dataset (the parts of it used in this project), then `ecp_dataset_setup.sh` to set up the dataset and format it for model training.

## Running the model
### Virtual environment
Use a [virtual environment](https://docs.python.org/3/library/venv.html) if necessary to install the ultralytics package. This will likely be necessary on a remote server, because you won't have permission to edit the server environment.
1. `python -m venv` to create a new virtual environment
2. `source .venv/bin/activate` to activate
3. `pip install ultralytics` to install the ultralytics package (after activating the virtual environment)
4. Run any code that imports the ultralytics package while the virtual environment is activated
5. `deactivate` to deactivate (it can be activated again at any point)

### tmux
Use [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) if running on a remote machine, so that training (and/or other time-consuming tasks) doesn't get interrupted by your computer losing power or connection to the server. [Screen](https://www.gnu.org/software/screen/manual/screen.html) is also an option, but it is a little more awkward to use.
1. `tmux new` to create a new session
2. Start running some time-consuming command (for example, `python3 train.py`)
3. `ctrl+b then press d` to detach from the session
4. `tmux attach` to re-attach and check in later

### Overall process
Setup:
1. Create a virtual environment if needed
2. `source .venv/bin/activate` to activate that environment
3. `pip install ultralytics`
Also, make sure that if you're on a remote machine, everything is stored in some directory that is local to that machine, not just on a server. Needing to wait for network traffic will make your model much slower to run.

Training the model:
1. `tmux new` to create a new tmux session
2. `source .venv/bin/activate` to activate your virtual environment
3. `python3 train.py`
4. `ctrl+b then press d` to detach from the tmux session
5. `tmux attach` to check in later

### Image cache options
`cache` is a [parameter](https://docs.ultralytics.com/usage/cfg/#train-settings) in YOLO's train model function that enables image caching, which speeds up training at the cost of additional memory usage. Its default value is False. In this code it is set to `True` to cache images using RAM. The other option is to set it to `'disk'` to cache images by generating and saving numpy array files where the dataset images are. The array files are about 5x the size of each image, if I'm remembering correctly. This didn't work well for my setup, since I had more limits on disk storage than on RAM, especially considering the size of the ECP dataset. However, if you have limited RAM usage but a very high storage limit, then setting it to `'disk'` would be better. If you're limited on both, `cache=False` will work, it will just be slower.
