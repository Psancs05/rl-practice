import os
import gym
import cv2
import wandb
import time
from tqdm import trange
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from DDQN import DDQNAgent
from ddqn_t import DDQNTAgent
from src.wrappers import apply_wrappers


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = COMPLEX_MOVEMENT    


def main():
    # Define the environment
    env_name = "SuperMarioBrosRandomStages-v0"
    env = gym.make(env_name, stages = ["1-1", "1-2", "1-3", "1-4"], apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH)


    # ========================== Hiperparameters ==========================
    max_episodes = 100_000_000              # max episodes
    
    lr = 1e-4                               # learning rate
    gamma = 0.999                           # discount factor
    epsilon = 1.0                           # exploration factor

    eps_decay = 0.9999                      # epsilon decay
    eps_min = 0.05                          # minimum epsilon
    replay_buffer_capacity = 40_000         # replay buffer capacity
    batch_size = 256                        # batch size
    sync_network_rate = 10_000              # sync network rate

    save_model_episodes = 10_000           # save model every n episodes
    checkpoint_base_path = "model_checkpoints/ddqn/ddqn_model"

    log_info_episodes = 10                  # log model info every n episodes

    # ========================== Environment ==========================
    state, _ = env.reset()
    done = False

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]

    # ========================== Model ==========================
    ddqn_agent = DDQNTAgent(input_channels, action_dim, lr, gamma, epsilon, eps_decay, eps_min, replay_buffer_capacity, batch_size, sync_network_rate, (IMG_HEIGHT, IMG_WIDTH))
    os.makedirs('model_checkpoints/ddqn', exist_ok=True)

    # ========================== WandB ==========================
    wandb.init(project="mario-ddqn-final", sync_tensorboard=False)
    wandb.require("core")
    model_id = wandb.run.id

    # add hyperparameters to wandb
    wandb.config.lr = lr
    wandb.config.gamma = gamma
    wandb.config.epsilon = epsilon
    wandb.config.eps_decay = eps_decay
    wandb.config.eps_min = eps_min
    wandb.config.replay_buffer_capacity = replay_buffer_capacity
    wandb.config.batch_size = batch_size
    wandb.config.sync_network_rate = sync_network_rate
    wandb.config.save_model_episodes = save_model_episodes
    wandb.config.log_info_episodes = log_info_episodes

    # ========================== Training ==========================
    episode_num = 1
    episode_reward_mean = 0
    total_reward = 0
    total_time = 0

    while episode_num < max_episodes:
        state, _ = env.reset()
        done = False
        current_ep_reward = 0
        episode_actions = []

        start_ep_time = time.time()

        while not done:
            action = ddqn_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            current_ep_reward += reward

            ddqn_agent.add_to_buffer(state, action, reward, next_state, done)

            state = next_state
            episode_actions.append(action)

            if done:
                break

        # Update agent
        ddqn_agent.update()

        end_ep_time = time.time()
        time_per_episode = end_ep_time - start_ep_time
        total_time += end_ep_time - start_ep_time
        time_avg_per_episode = total_time / episode_num

        total_reward += current_ep_reward
        episode_reward_mean = total_reward / (episode_num + 1)
        episode_num += 1
       
        # Save model
        if (episode_num + 1) % save_model_episodes == 0:
            print(f"----- Saving model at episode {episode_num} -----")
            ddqn_agent.save(f"{checkpoint_base_path}_{episode_num}_{model_id}_iter.pth") 

            wandb.log({"episode_actions": wandb.Histogram(episode_actions)})    

        # Log model info
        if (episode_num + 1) % log_info_episodes == 0:
            print(f"Episode: {episode_num + 1}, Total reward: {current_ep_reward}, Mean reward: {episode_reward_mean}, Epsilon: {ddqn_agent.epsilon}")
        
        wandb.log({
            "episode_reward": current_ep_reward,
            "episode_reward_mean": episode_reward_mean,
            "epsilon": ddqn_agent.epsilon,
            "episode_num": episode_num,
            "time_per_episode": time_per_episode,
            "time_avg_per_episode": time_avg_per_episode
        })    

    env.close()
    cv2.destroyAllWindows()
    wandb.finish()


if __name__ == "__main__":
    main()