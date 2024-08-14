import os
import gym
import cv2
import wandb
import time
from tqdm import tqdm
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from DDQN import DDQNAgent
from src.wrappers import apply_wrappers


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = SIMPLE_MOVEMENT    


def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH, skip=4, stack=2)


    # ========================== Hiperparameters ==========================
    max_episodes = 400_000             # max episodes
    lr = 0.00025                       # learning rate
    gamma = 0.99                       # discount factor

    epsilon = 1.0                      # exploration factor
    eps_decay = 5**4                  # epsilon decay
    eps_min = 0.05                     # minimum epsilon

    replay_buffer_capacity = 40_000    # replay buffer capacity
    batch_size = 32                    # batch size
    sync_network_rate = 10_000         # sync network rate
    update_steps = 4                   # update model every n steps
    min_exp = 10_000                    # minimum experiences to start training

    save_model_episodes = 2_000        # save model every n episodes
    checkpoint_base_path = "model_checkpoints/ddqn/ddqn_model"

    log_info_episodes = 10             # log model info every n episodes
    log_movements_episodes = 1_000     # log movements every n episodes

    # ========================== Environment ========================== 
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape

    # ========================== Model ==========================
    ddqn_agent = DDQNAgent(
        input_shape=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        eps_decay=eps_decay,
        eps_min=eps_min,
        bs=batch_size,
        sync_network_steps=sync_network_rate,
        replay_buffer_size=replay_buffer_capacity
    )
    
    os.makedirs('model_checkpoints/ddqn', exist_ok=True)

    # ========================== WandB ==========================
    wandb.init(project="mario-rl-ddqn-git", sync_tensorboard=False)
    wandb.require("core")
    model_id = wandb.run.id

    wandb.config.total_episodes = max_episodes
    wandb.config.learning_rate = lr
    wandb.config.discount_factor = gamma
    wandb.config.epsilon = epsilon
    wandb.config.epsilon_decay = eps_decay
    wandb.config.epsilon_start = epsilon
    wandb.config.epsilon_min = eps_min
    wandb.config.replay_buffer_capacity = replay_buffer_capacity
    wandb.config.batch_size = batch_size
    wandb.config.sync_network_rate = sync_network_rate
    wandb.config.update_steps = update_steps
    wandb.config.min_exp = min_exp
    wandb.config.save_model_episodes = save_model_episodes
    wandb.config.checkpoint_base_path = checkpoint_base_path
    wandb.config.log_info_episodes = log_info_episodes
    wandb.config.log_movements_steps = log_movements_episodes

    # ========================== Training ==========================
    total_steps = 0
    total_reward = 0
    total_time = 0
    episode_reward_mean = 0
    episode_time_mean = 0

    for ep in tqdm(range(max_episodes)):

        state, info = env.reset()
        done = False
        q, loss = None, None

        episode_reward = 0
        episode_time = 0
        episode_actions = []
        start_ep_time = time.time()

        epsilon = ddqn_agent.get_epsilon(ep)
        ddqn_agent.epsilon = epsilon

        while not done:
            total_steps += 1

            action = ddqn_agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            episode_actions.append(action)
            episode_reward += reward

            ddqn_agent.add_to_buffer(state, action, reward, next_state, done)
            state = next_state

            # First min_exp are not collected
            if len(ddqn_agent.replay_buffer) < min_exp:
                continue

            # Sync networks
            if total_steps % sync_network_rate == 0:
                ddqn_agent.sync_networks()

            # Update model
            if total_steps % update_steps == 0:
                q, loss = ddqn_agent.update()
            else:
                q, loss = None, None

            # Save model
            if ep % save_model_episodes == 0:
                ddqn_agent.save(f"{checkpoint_base_path}_{total_steps}_{model_id}_iter.pth")

        # Log episode actions
        if ep % log_movements_episodes == 0:
            wandb.log({"episode_actions": wandb.Histogram(episode_actions)})

        # Stats
        end_ep_time = time.time()

        episode_time += end_ep_time - start_ep_time
        total_time += episode_time
        episode_time_mean = total_time / (ep + 1)

        total_reward += episode_reward
        episode_reward_mean = total_reward / (ep + 1)


        wandb.log({
            "episode_reward": episode_reward,
            "epsilon": epsilon,
            "episode": ep,
            "q_value": q,
            "loss": loss,
            "distance": info['x_pos'],
            "episode_time": episode_time,
            "episode_time_mean": episode_time_mean,
            "episode_reward": episode_reward,
            "episode_reward_mean": episode_reward_mean
        })


    env.close()
    cv2.destroyAllWindows()
    wandb.finish()


if __name__ == "__main__":
    main()