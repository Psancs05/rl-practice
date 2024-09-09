# rl-practice
Repo to practice RL algorithms

# Algorithms
## PPO
To run PPO agent execute `python src/ppo/train.py`
Hyperparameters are defined in train.py


## DDQN
To run DDQN agent execute `python src/ddqn/train.py`
Hyperparameters are defined in train.py

## Test
To run a single episode with render mode on run `python src/test.py`
Inside the file, you will have to define the agent to use and the checkpoint path with the weights

# Installation
* Install dependencies `pip install -r requirements.txt`
* Run `apt install libgl1-mesa-glx`
* (Optional) Add project to pythonpath `export PYTHONPATH=$(pwd)` 


# Demos

## DDQN
![](https://github.com/jsancs/rl-practice/blob/main/demos/ddqn.gif)

## PPO
![](https://github.com/jsancs/rl-practice/blob/main/demos/ppo.gif)

## DT1
![](https://github.com/jsancs/rl-practice/blob/main/demos/dt1.gif)

## DT2
![](https://github.com/jsancs/rl-practice/blob/main/demos/dt2.gif)
