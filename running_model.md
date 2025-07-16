Instructions for training the model:

Setup:
1. Make a virtual environment:  `python -m venv`
2. In that virtual environment: `pip install ultralytics`
3. While connected to a specific machines, copy dataset from /stash/portal/eurocity to /disk/local/scratch
4. Copy over code directory there as well, including virtual environment
5. Make sure the code and dataset directories are both under some parent directory, following the file structure in the README

Running it:
1. Directory to run from: `/disk/local/scratch/project/code` (`code` being this repository)
2. Start a new tmux session: `tmux new`
3. Activate the virtual environment: `source .venv/bin/activate`
4. run `python3 train.py`
5. Detach tmux session: ctrl+b then press d
6. Re-attach tmux session: `tmux attach`
7. Deactivate virtual environment: `deactivate`
