import gym
import cv2
import time
import os

import wandb
from tqdm import trange

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from src.wrappers import apply_wrappers
from PPO import PPO


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = SIMPLE_MOVEMENT
        

def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"

    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH, skip=4, stack=2)


    # ========================== Hiperparameters ==========================
    # Define hiperparámetros
    input_channels = 2  # Usamos 4 stacks de frames
    action_dim = len(SIMPLE_MOVEMENT)
    img_dim = (84, 84)
    lr = 2.5e-4
    gamma = 0.99
    eps_clip = 0.2
    k_epochs = 4
    total_timesteps = 1_000_000_000
    update_timestep = 2048
    # gae_lambda = 0.95

    # max_training_steps = int(5e6)     

    # # update_episode = 5                   
    # update_steps = 2048
    # K_epochs = 5                          

    # eps_clip = 0.2                         
    # gamma = 0.99                          

    # lr_actor = 0.0001                       
    # lr_critic = 0.0001              

    save_model_steps = 100_000          
    checkpoint_base_path = "model_checkpoints/ppo/ppo_model"

    log_info_steps = 1000       
    log_movements_steps = 50_000


    # ========================= Enviroment ==========================
    # action_dim = env.action_space.n
    # state_dim = env.observation_space.shape
    # input_channels = state_dim[0]


    # ========================== Initial Hyperparameters ==========================
    # print("------- Initial Hyperparameters -------")

    # print("Action Dimension: ", action_dim)
    # print("State Dimension: ", state_dim)
    # print("Input channels: ", input_channels)

    # print("Environment: ", env_name)
    # print("Max training episodes: ", max_training_episodes)
    # # print("Update policy every n episodes: ", update_episode)
    # print("Update policy every n steps: ", update_steps)
    # print("Update policy for K epochs in one PPO update: ", K_epochs)
    # print("Clip parameter for PPO: ", eps_clip)
    # print("Discount factor: ", gamma)
    # print("Learning rate for actor network: ", lr_actor)
    # print("Learning rate for critic network: ", lr_critic)

    # print("--------------------------------------\n")

    # ========================== Model ==========================
    # ppo_agent = PPO(input_channels, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, img_dim=(IMG_HEIGHT, IMG_WIDTH))
    ppo_agent = PPO(input_channels, action_dim, lr, gamma, k_epochs, eps_clip, img_dim)
    os.makedirs('model_checkpoints/ppo', exist_ok=True)

    # Get model summary
    total_params = sum(p.numel() for p in ppo_agent.policy_old.parameters())
    print(f'Total number of parameters: {total_params}')

    # ========================== WandB ==========================
    wandb.init(project="mario-ppo-final", sync_tensorboard=False)
    wandb.require("core")
    model_id = wandb.run.id

    # add hyperparameters to wandb
    # wandb.config.lr_actor = lr_actor
    # wandb.config.lr_critic = lr_critic
    # wandb.config.gamma = gamma
    # wandb.config.K_epochs = K_epochs
    # wandb.config.eps_clip = eps_clip
    # wandb.config.max_training_episodes = max_training_episodes
    # # wandb.config.update_episode = update_episode
    # wandb.config.update_steps = update_steps
    # wandb.config.save_model_episodes = save_model_episodes
    # wandb.config.log_info_episodes = log_info_episodes
    # wandb.config.total_params = total_params

    wandb.watch(ppo_agent.policy_old)
    wandb.watch(ppo_agent.policy)

    wandb.config.update_timestep = update_timestep
    wandb.config.total_timesteps = total_timesteps
    wandb.config.save_model_steps = save_model_steps
    wandb.config.log_info_steps = log_info_steps
    wandb.config.log_movements_steps = log_movements_steps
    wandb.config.lr = lr
    wandb.config.gamma = gamma
    wandb.config.eps_clip = eps_clip
    wandb.config.k_epochs = k_epochs
    wandb.config.img_height = IMG_HEIGHT
    wandb.config.img_width = IMG_WIDTH
    wandb.config.movement = MOVEMENT

    # # ========================== Trainig ==========================
    # episode_num = 1
    # episode_reward_mean = 0
    # total_reward = 0
    # total_time = 0
    # total_steps = 0

    # # while episode_num <= max_training_episodes:
    # while total_steps <= max_training_steps:
    #     state, _ = env.reset()
    #     current_ep_reward = 0
    #     done = False
    #     episode_actions = []

    #     start_ep_time = time.time()

    #     while not done:
    #         action = ppo_agent.select_action(state)
    #         next_state, reward, done, _, __ = env.step(action)

    #         ppo_agent.buffer.rewards.append(reward)
    #         ppo_agent.buffer.is_terminals.append(done)

    #         current_ep_reward += reward
    #         episode_actions.append(action)
            
    #         total_steps += 1
    #         state = next_state

    #         # update PPO agent
    #         if total_steps % update_steps == 0:
    #             loss = ppo_agent.update()
    #             print(f"Loss: {loss}")

    #         if done:
    #             break

    #     end_ep_time = time.time()
    #     time_per_episode = end_ep_time - start_ep_time
    #     total_time += time_per_episode
    #     time_avg_per_episode = total_time / episode_num

    #     # save model checkpoint
    #     if episode_num % save_model_episodes == 0:
    #         print(f"------ Saving model checkpoint at episode {episode_num} ------")
    #         ppo_agent.save(f"{checkpoint_base_path}_{episode_num}_{model_id}.pth")

    #     # # log movements
    #     # if episode_num % log_movements_episodes == 0:
    #     #     wandb.log({"episode_actions": wandb.Histogram(episode_actions)})

    #     # log model info
    #     if episode_num % log_info_episodes == 0:
    #         print(f"Model Info: Episode {episode_num}, Reward: {current_ep_reward}, Mean Reward: {episode_reward_mean}")

    #     total_reward += current_ep_reward
    #     episode_reward_mean = total_reward / episode_num
    #     episode_num += 1


    #     # wandb.log({
    #     #     "episode_reward": current_ep_reward,
    #     #     "episode_reward_mean": episode_reward_mean,
    #     #     "episode_num": episode_num,
    #     #     "time_per_episode": time_per_episode,
    #     #     "time_avg_per_episode": time_avg_per_episode
    #     # })

    # env.close()
    # cv2.destroyAllWindows()
    # # wandb.finish()


    timestep = 0
    episode_num = 1
    episode_reward_mean = 0
    total_reward = 0
    total_time = 0

    for _ in range(total_timesteps // update_timestep):
    # while True:
        state, _ = env.reset()
        done = False
        episode_actions = []
        episode_reward = 0
        episode_time = 0

        start_time = time.time()
        while not done:
            action, action_logprob = ppo_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            # store transition
            ppo_agent.store_transition(state, action, action_logprob, reward, done)

            # update state
            state = next_state
            timestep += 1

            # update episode reward
            episode_reward += reward
            episode_actions.append(action)

            # update PPO agent
            if timestep % update_timestep == 0:
                loss = ppo_agent.update()
                wandb.log({"loss": loss})

            if done:
                break

            # Save model
            if timestep % save_model_steps == 0:
                print(f"Saving model at timestep {timestep}")
                ppo_agent.save(f"{checkpoint_base_path}_{timestep}_{model_id}.pth")

            # Log info
            if timestep % log_info_steps == 0:
                print(f"Model Info: Timestep {timestep}, Reward: {episode_reward}, Mean Reward: {episode_reward_mean}")
        
            # Log movements
            if timestep % log_movements_steps == 0:
                wandb.log({"episode_actions": wandb.Histogram(episode_actions)})

        end_time = time.time()
        episode_time = end_time - start_time
        total_time += episode_time
        episode_time_avg = total_time / episode_num

        total_reward += episode_reward
        episode_reward_mean = total_reward / episode_num

        episode_num += 1

        # Log episode info
        wandb.log({
            "episode_reward": episode_reward,
            "episode_reward_mean": episode_reward_mean,
            "episode_num": episode_num,
            "episode_time": episode_time,
            "episode_time_avg": episode_time_avg
        })

    env.close()
    cv2.destroyAllWindows()
    wandb.finish()


if __name__ == "__main__":
    main()
