import gym
import cv2
import os

import wandb
from tqdm import trange

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from src.wrappers import apply_wrappers
from PPO import PPO


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = COMPLEX_MOVEMENT
        

def main():
    # Define the environment
    env_name = "SuperMarioBrosRandomStages-v0"

    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH, 4, 4)


    # ========================== Hiperparameters ==========================

    max_ep_len = 500                      # max timesteps in one episode
    max_training_timesteps = 100_000       # break training loop if timeteps > max_training_timesteps

    update_timestep = max_ep_len * 5        # update policy every n timesteps
    K_epochs = 80                           # update policy for K epochs in one PPO update

    eps_clip = 0.2                          # clip parameter for PPO    
    gamma = 0.99                            # discount factor

    lr_actor = 0.0003                       # learning rate for actor network
    lr_critic = 0.001                       # learning rate for critic network

    save_model_steps = 100_000            # save model every n timesteps
    checkpoint_base_path = "model_checkpoints/ppo/ppo_model"

    log_info_steps = 5000                    # log model info every n timesteps

    # ========================= Enviroment ==========================
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]


    # ========================== Initial Hyperparameters ==========================
    print("------- Initial Hyperparameters -------")

    print("Action Dimension: ", action_dim)
    print("State Dimension: ", state_dim)
    print("Input channels: ", input_channels)

    print("Max training timesteps: ", max_training_timesteps)
    print("Max timesteps in one episode: ", max_ep_len)
    print("Update policy every n timesteps: ", update_timestep)
    print("Update policy for K epochs in one PPO update: ", K_epochs)
    print("Clip parameter for PPO: ", eps_clip)
    print("Discount factor: ", gamma)
    print("Learning rate for actor network: ", lr_actor)
    print("Learning rate for critic network: ", lr_critic)

    print("--------------------------------------\n")

    # ========================== Model ==========================
    ppo_agent = PPO(input_channels, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, img_dim=(IMG_HEIGHT, IMG_WIDTH))
    os.makedirs('model_checkpoints/ppo', exist_ok=True)

    # Get model summary
    total_params = sum(p.numel() for p in ppo_agent.policy_old.parameters())
    print(f'Total number of parameters: {total_params}')

    # ========================== WandB ==========================
    wandb.init(project="mario-ppo", sync_tensorboard=True)
    wandb.require("core")

    # add hyperparameters to wandb
    wandb.config.lr_actor = lr_actor
    wandb.config.lr_critic = lr_critic
    wandb.config.gamma = gamma
    wandb.config.K_epochs = K_epochs
    wandb.config.eps_clip = eps_clip
    wandb.config.max_training_timesteps = max_training_timesteps
    wandb.config.max_ep_len = max_ep_len
    wandb.config.update_timestep = update_timestep
    wandb.config.save_model_steps = save_model_steps
    wandb.config.log_info_steps = log_info_steps

    # ========================== Trainig ==========================
    time_step = 0
    episode_num = 1
    episode_reward_mean = 0
    total_reward = 0

    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0
        episode_actions = []

        for _ in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _, __ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            episode_actions.append(action)
            
            if done:
                break

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update() 

            # save model checkpoint
            if time_step % save_model_steps == 0:
                print(f"------ Saving model checkpoint at timestep {time_step} ------")
                model_id = wandb.run.id
                ppo_agent.save(f"{checkpoint_base_path}_{episode_num}_{model_id}.pth")

            # log model info
            if time_step % log_info_steps == 0:
                print("------ Model Info ------")
                print(f"Time step: {time_step}")
                print(f"Episode: {episode_num}")
                print(f"Current Episode Reward: {current_ep_reward}")
                print(f"Episode Reward Mean: {episode_reward_mean}")
                print()

        episode_num += 1
        total_reward += current_ep_reward
        episode_reward_mean = total_reward / episode_num

        wandb.log({"episode_reward": current_ep_reward}, step=episode_num)
        wandb.log({"episode_reward_mean": episode_reward_mean}, step=episode_num)
        wandb.log({"time_step": time_step}, step=episode_num)
        wandb.log({"actions": wandb.Histogram(episode_actions)}, step=episode_num)


    env.close()
    cv2.destroyAllWindows()
    wandb.finish()


if __name__ == "__main__":
    main()
