Instructions for training the model:
1. `cd` to whatever directory the repository is
2. Start a new tmux session: `tmux new -smy_session`
3. Activate the virtual environment: `source .venv/bin/activate`
4. run `python3 train.py`
5. Detach tmux session: ctrl+b then press d
6. Re-attach tmux session: `tmux attach`
7. Deactivate virtual environment: `deactivate`
