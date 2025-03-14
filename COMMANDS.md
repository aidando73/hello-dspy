```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./env python=3.10
source ~/miniconda3/bin/activate && conda activate ./env
pip install -r requirements.txt

mlflow ui --port 5000

# Persistent jupyter lab session
tmux new -s jupyter_session
source ~/miniconda3/bin/activate && conda activate ./env
conda install -c anaconda ipykernel
jupyter lab --ip 0.0.0.0 --port 8080 --no-browser --allow-root

# Add in kernel
pip install ipykernel
python -m ipykernel install --user --name=dspy_env
# Then set the kernel in 
```
