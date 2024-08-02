# rl-practice
Repo to practice RL algorithms

# Algorithms
## PPO
To run PPO agent execute `python ppo/train.py`
Hyperparameters are defined in train.py


## DDQN
To run DDQN agent execute `python ddqn/train.py`
Hyperparameters are defined in train.py

## Test
To run a single episode with render mode on run `python tes.py`
Inside the file, you will have to define the agent to use and the checkpoint path with the weights

# Installation
* Install dependencies `pip install -r requirements.txt`
* Run `apt install libgl1-mesa-glx`