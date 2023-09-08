# Project Details

This project trains an agent to interact with Udacity's Bananas World such that it learns to pickup the yellow banans (+1) and ignore the blue bananas (-1).

The code is written in PyTorch and Python 3.

## Getting Started

After following the instructions defined here for downloading and installing: https://github.com/udacity/deep-reinforcement-learning/tree/master

My installation was based on 
* Windows 11 x64
* Python 3.6.13 :: Anaconda, Inc.

```bash
# after creating new conda python environment
cd <path/to/dev/directory>
git clone https://github.com/udacity/deep-reinforcement-learning.git
git clone https://github.com/PHRupp/udacity-agent-is-bananas.git

# used a specific version of torch
pip install https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install matplotlib==3.3.4, numpy==1.19.5
```

## Usage

In order to run the code

```python
python main.py
```

