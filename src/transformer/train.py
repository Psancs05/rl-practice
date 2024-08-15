import gym
import torch
import wandb
import time
import os
from tqdm import trange
from src.transformer.model_imp import DTAgent
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
    n_episodes = 1_000_000
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

    update_timestep = 16
    save_model_episodes = 1_000
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

    agent.model.train()

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False

        episode_actions = []
        episode_states = []
        episode_rewards = []
        episode_timesteps = []
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        episode_states.append(state)
        steps = 0
        current_episode_reward = 0
        current_episode_actions = []

        start_time = time.time()
        while not done:
            episode_timesteps.append(steps)
            
            if len(episode_states) <= context_len:
                # add padding to the states
                padded_states = [torch.zeros_like(state) for _ in range(context_len - len(episode_states))] + episode_states
                actions_tensor = torch.tensor([0] * (context_len - len(episode_actions)) + episode_actions, dtype=torch.long).unsqueeze(0).to(device)
                returns_to_go = torch.tensor([0.0] * (context_len - len(episode_rewards)) + episode_rewards, dtype=torch.float32).unsqueeze(0).to(device)
                timesteps_tensor = torch.tensor([0] * (context_len - len(episode_timesteps)) + episode_timesteps, dtype=torch.long).unsqueeze(0).to(device)
            else:
                padded_states = episode_states[-context_len:]
                actions_tensor = torch.tensor(episode_actions[-context_len:], dtype=torch.long).unsqueeze(0).to(device)
                returns_to_go = torch.tensor(episode_rewards[-context_len:], dtype=torch.float32).unsqueeze(0).to(device)
                timesteps_tensor = torch.tensor(episode_timesteps[-context_len:], dtype=torch.long).unsqueeze(0).to(device)

            # convert the states to a tensor
            states_tensor = torch.cat(padded_states).unsqueeze(0)
            
            # select the action
            action = agent.select_action(
                states_tensor,
                actions_tensor,
                returns_to_go,
                timesteps_tensor
            )
            episode_actions.append(action)
            current_episode_actions.append(action)
            
            next_state, reward, done, _, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            episode_states.append(next_state)
            episode_rewards.append(reward)
            
            # store the transition
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
            current_episode_reward += reward
            
            # update the model
            if steps % update_timestep == 0:
                loss = agent.update(batch_size, gamma)
                wandb.log({"loss": loss})
            
            if done:
                break
        
        end_time = time.time()
        episode_time = end_time - start_time
        total_time += episode_time
        episode_time_avg = total_time / (episode + 1)

        total_reward += current_episode_reward
        episode_reward_mean = total_reward / (episode + 1)

        # log movements
        if episode % log_movements_episodes == 0:
            wandb.log({"episode_actions": wandb.Histogram(current_episode_actions)})

        wandb.log({
            "steps": steps,
            "distance": info["x_pos"],
            "episode_time": episode_time,
            "episode_time_avg": episode_time_avg,
            "episode_reward": current_episode_reward,
            "episode_reward_mean": episode_reward_mean
        })
        
        print(f'Episode {episode+1}/{n_episodes}, Reward: {current_episode_reward}, Distance: {info["x_pos"]}')

        # save the model
        if (episode+1) % save_model_episodes == 0:
            print(f"Saving model at episode {episode+1}")
            agent.save(f"{checkpoint_base_path}_{episode+1}_{model_id}.pth")
        
    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
