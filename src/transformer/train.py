import gym
import torch
import wandb
import time
import os
from tqdm import trange
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

    log_movements_episodes = 50
    checkpoint_base_path = "model_checkpoints/dt/dt_model"

    # ========================== Model ==========================
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
    os.makedirs('model_checkpoints/dt', exist_ok=True)

    # get model summary
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f'Total number of parameters: {total_params}')

    # ========================== WandB ==========================
    wandb.init(project="mario-dt-final", sync_tensorboard=False)
    wandb.require("core")
    model_id = wandb.run.id

    wandb.watch(agent.model)

    wandb.config.n_episodes = n_episodes
    wandb.config.batch_size = batch_size
    wandb.config.gamma = gamma
    wandb.config.state_dim = state_dim
    wandb.config.act_dim = act_dim
    wandb.config.skip_interval = skip_interval
    wandb.config.buffer_size = buffer_size
    wandb.config.context_len = context_len
    wandb.config.n_blocks = n_blocks
    wandb.config.h_dim = h_dim
    wandb.config.n_heads = n_heads
    wandb.config.drop_p = drop_p


    # ========================== Trainig ==========================

    steps = 0
    episode_reward_mean = 0
    total_reward = 0
    total_time = 0

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False

        epissode_actions = []
        episode_reward = 0

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        actions = torch.zeros((1, context_len), dtype=torch.long).to(device)
        returns_to_go = torch.zeros((1, context_len, 1), dtype=torch.float32).to(device)
        timesteps = torch.zeros((1, context_len), dtype=torch.long).to(device)
        
        start_time = time.time()
        while not done:
            action = agent.select_action(state, actions, returns_to_go, timesteps)
            epissode_actions.append(action)
            
            next_state, reward, done, _, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # add the transition to the replay buffer
            if steps % skip_interval == 0:
                agent.store_transition(state, action, reward, next_state, done)

            state = next_state

            episode_reward += reward
            steps += 1
            
            if done:
                break

        # update the model
        loss = agent.update(batch_size, gamma, timesteps)

        
        end_time = time.time()
        episode_time = end_time - start_time
        total_time += episode_time
        episode_time_avg = total_time / (episode + 1)

        total_reward += episode_reward
        episode_reward_mean = total_reward / (episode + 1)

        # log movements
        if episode % log_movements_episodes == 0:
            wandb.log({"episode_actions": wandb.Histogram(epissode_actions)})

        wandb.log({
            "loss": loss,
            "steps": steps,
            "episode_time": episode_time,
            "episode_time_avg": episode_time_avg,
            "episode_reward": episode_reward,
            "episode_reward_mean": episode_reward_mean
        })
        

        
        
        print(f'Episode {episode+1}/{n_episodes}, Reward: {episode_reward}')

    # save the model
    agent.save(f"{checkpoint_base_path}_{episode+1}_{model_id}.pth")
        
    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
