import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from collections import deque
from model import DecisionTransformer
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.wrappers import apply_wrappers


################################## set device ##################################
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
    learning_rate = 1e-4
    gamma = 0.99  # Factor de descuento
    state_dim = env.observation_space.shape[0]
    act_dim = len(SIMPLE_MOVEMENT)

    # model hiperparameters
    context_len = 10
    n_blocks = 4
    h_dim = 128
    n_heads = 4
    drop_p = 0.1

    # create the model and optimizer
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=drop_p
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # get model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    # Replay buffer
    replay_buffer = deque(maxlen=10000)

    def select_action(state, actions, returns_to_go, timesteps):
        with torch.no_grad():
            action_logits = model(state, actions, returns_to_go, timesteps)
            action_prob = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_prob, num_samples=1)
        return action.item()

    # training loop
    start_time = time.time()
    for episode in range(n_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        actions = torch.zeros((1, context_len), dtype=torch.long).to(device)
        returns_to_go = torch.zeros((1, context_len, 1), dtype=torch.float32).to(device)
        timesteps = torch.zeros((1, context_len), dtype=torch.long).to(device)
        
        episode_reward = 0
        for t in range(max_steps_per_episode):
            action = select_action(state, actions, returns_to_go, timesteps)
            
            next_state, reward, done, _, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            episode_reward += reward
            
            # add the transition to the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            
            state = next_state
            
            if done:
                break
        
        # train the model
        if len(replay_buffer) >= batch_size:
            batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
            
            states, actions, rewards, next_states, dones = zip(*[replay_buffer[idx] for idx in batch])
            
            states = torch.cat(states).to(device)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            next_states = torch.cat(next_states).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
            
            returns_to_go = torch.zeros_like(rewards).to(device)
            future_return = 0.0
            for i in reversed(range(rewards.size(0))):
                future_return = rewards[i] + gamma * future_return * (1 - dones[i])
                returns_to_go[i] = future_return
            
            # forward pass
            action_preds = model(states, actions, returns_to_go, timesteps)
            
            # compute the loss
            loss = F.cross_entropy(action_preds, actions.squeeze(-1))
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Episode {episode+1}/{n_episodes}, Reward: {episode_reward}')


    end_time = time.time()
    print(f'Training took: {end_time - start_time} seconds')

    # save the model
    torch.save(model.state_dict(), 'decision_transformer.pth')
        
    env.close()


if __name__ == '__main__':
    main()
