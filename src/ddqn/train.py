import os
import gym
import cv2
import wandb

from tqdm import trange
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from DDQN import DDQNAgent
from src.wrappers import apply_wrappers


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = COMPLEX_MOVEMENT    


def main():
    # Define the environment
    env_name = "SuperMarioBrosRandomStages-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH)


    # ========================== Hiperparameters ==========================
    max_episodes = 750                      # max timesteps in one episode
    
    lr = 0.00025                            # learning rate
    gamma = 0.9                             # discount factor
    epsilon = 1.0                           # exploration factor

    eps_decay = 0.99999975                  # epsilon decay
    eps_min = 0.05                          # minimum epsilon
    replay_buffer_capacity = 100_000        # replay buffer capacity
    batch_size = 32                         # batch size
    sync_network_rate = 10_000              # sync network rate

    save_model_episodes = 750               # save model every n episodes
    checkpoint_base_path = "model_checkpoints/ddqn/ddqn_model"

    log_info_episodes = 10                  # log model info every n episodes

    # ========================== Environment ==========================
    state, _ = env.reset()
    done = False

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]


    # ========================== Model ==========================
    ddqn_agent = DDQNAgent(input_channels, action_dim, lr, gamma, epsilon, eps_decay, eps_min, replay_buffer_capacity, batch_size, sync_network_rate, (IMG_HEIGHT, IMG_WIDTH))
    os.makedirs('model_checkpoints/ddqn', exist_ok=True)

    # Get model summary
    total_params = sum(p.numel() for p in ddqn_agent.online_network.parameters())
    print(f'Total number of parameters: {total_params}')

    # ========================== WandB ==========================
    wandb.init(project="mario-ddqn", sync_tensorboard=False)
    wandb.require("core")

    wandb.watch(ddqn_agent.online_network)
    wandb.watch(ddqn_agent.target_network)

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
    episode_num = 0
    max_ep_len = 500
    episode_reward_mean = 0
    total_reward = 0

    while episode_num < max_episodes:
        state, _ = env.reset()
        done = False
        current_ep_reward = 0
        episode_actions = []

        for _ in range(1, max_ep_len+1):
            # time.sleep(0.5)
            action = ddqn_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            current_ep_reward += reward

            ddqn_agent.add_to_buffer(state, action, reward, next_state, done)
            ddqn_agent.update()

            state = next_state
            episode_actions.append(action)

            if done:
                break

        total_reward += current_ep_reward
        episode_reward_mean = total_reward / (episode_num + 1)

        wandb.log({
            "Episode": episode_num,
            "episode_reward": current_ep_reward,
            "episode_reward_mean": episode_reward_mean,
            "epsilon": ddqn_agent.epsilon,
            "actions": wandb.Histogram(episode_actions)
        })
        # wandb.log({})
        # wandb.log({})
        # wandb.log({})
        # wandb.log({})      

        episode_num += 1
        # print(f"Reward: {current_ep_reward}")
       
        # Save model
        if (episode_num + 1) % save_model_episodes == 0:
            print(f"----- Saving model at episode {episode_num + 1} -----")
            ddqn_agent.save(f"{checkpoint_base_path}_{episode_num + 1}_iter.pth")     

        # Log model info
        if (episode_num + 1) % log_info_episodes == 0:
            print(f"Episode: {episode_num + 1}, Total reward: {current_ep_reward}, Mean reward: {episode_reward_mean}, Epsilon: {ddqn_agent.epsilon}")

    env.close()
    cv2.destroyAllWindows()
    wandb.finish()


if __name__ == "__main__":
    main()