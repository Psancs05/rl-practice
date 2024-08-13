import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange
from collections import deque
from model import DTAgent
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.wrappers import apply_wrappers


print("==================================================================")
device = torch.device('cpu')
if torch.cuda.is_available(): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device set to : " + str(device))
else:
    print("Device set to : cpu")
print("==================================================================")


def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, SIMPLE_MOVEMENT, 84, 84, skip=4, stack=2)

    # ========================== Hiperparameters ==========================
    # training hiperparameters
    n_episodes = 1000
    max_steps_per_episode = 5000
    batch_size = 64
    gamma = 0.99
    state_dim = env.observation_space.shape[0]
    act_dim = len(SIMPLE_MOVEMENT)
    skip_interval = 4
    buffer_size = 10_000

    # model hiperparameters
    context_len = 20
    n_blocks = 4
    h_dim = 128
    n_heads = 4
    drop_p = 0.1

    # create the model and optimizer
    agent = DTAgent(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=drop_p,
        buffer_size=buffer_size,
    )

    # get model summary
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f'Total number of parameters: {total_params}')

    # training loop
    for episode in range(n_episodes):
        state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        actions = torch.zeros((1, context_len), dtype=torch.long).to(device)
        returns_to_go = torch.zeros((1, context_len, 1), dtype=torch.float32).to(device)
        timesteps = torch.zeros((1, context_len), dtype=torch.long).to(device)
        
        episode_reward = 0
        for t in trange(max_steps_per_episode):
            action = agent.select_action(state, actions, returns_to_go, timesteps)
            
            next_state, reward, done, _, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            episode_reward += reward
            
            # add the transition to the replay buffer
            if t % skip_interval == 0:
                agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            
            if done:
                break

        # update the model
        agent.update(batch_size, gamma, timesteps)
        
        
        print(f'Episode {episode+1}/{n_episodes}, Reward: {episode_reward}')

    # save the model
    agent.save(f"model_checpoints/decision_transformer/dt.pth")
        
    env.close()


if __name__ == '__main__':
    main()
