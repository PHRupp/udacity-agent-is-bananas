# Project Details

This project trains an agent to interact with Udacity's Bananas World such that it learns to pickup the yellow banans (+1) and ignore the blue bananas (-1).

The code is written in PyTorch and Python 3.

## Getting Started

After following the instructions defined here for downloading and installing: https://github.com/udacity/deep-reinforcement-learning/tree/master

My installation was based on 
* Windows 11 x64
* Python 3.6.13 :: Anaconda, Inc.

```bash
# Instructions from Deep RL course for environment setup
conda create --name drlnd python=3.6 
activate drlnd

# after creating new conda python environment
cd <path/to/dev/directory>
git clone https://github.com/udacity/deep-reinforcement-learning.git
git clone https://github.com/PHRupp/udacity-agent-is-bananas.git
pushd deep-reinforcement-learning/python

# HACK: edit requirements.txt to make "torch==0.4.1" since i couldnt get a working version for 0.4.0
pip install https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install .

# install packages used specifically in my code
pip install matplotlib==3.3.4, numpy==1.19.5
popd
pushd udacity-agent-is-bananas
```

## Usage

In order to run the code, we run it straight via python instead of using jupyter notebooks.

As depicted in the Report.pdf, you can change the paramters in dqn_agent.py and main.py to get different results. Otherwise, you can run the code as-is to get the same results assuming a random seed = 0. 

```python
# run from 'udacity-agent-is-bananas' directory
python main.py
```

