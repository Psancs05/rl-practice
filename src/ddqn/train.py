import os
import gym
import cv2
import wandb
import time
from tqdm import trange
from gym_super_mario_bros.actions import RIGHT_ONLY

from DDQN import DDQNAgent
from ddqn_t import DDQNTAgent
from src.wrappers import apply_wrappers


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = RIGHT_ONLY    


def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH, 4, 2)


    # ========================== Hiperparameters ==========================
    max_episodes = 400_000             # max episodes
    lr = 0.00025                       # learning rate
    gamma = 0.99                       # discount factor

    epsilon = 1.0                      # exploration factor
    eps_decay = 10**6                  # epsilon decay
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
    # wandb.init(project="mario-ddqn-final", sync_tensorboard=False)
    # wandb.require("core")
    # model_id = wandb.run.id

    # # add hyperparameters to wandb
    # wandb.config.lr = lr
    # wandb.config.gamma = gamma
    # wandb.config.epsilon = epsilon
    # wandb.config.eps_decay = eps_decay
    # wandb.config.eps_min = eps_min
    # wandb.config.replay_buffer_capacity = replay_buffer_capacity
    # wandb.config.batch_size = batch_size
    # wandb.config.sync_network_rate = sync_network_rate
    # wandb.config.save_model_episodes = save_model_episodes
    # wandb.config.log_info_episodes = log_info_episodes

    # ========================== Training ==========================
    total_steps = 0

    for _ in trange(max_episodes):

        state, _ = env.reset()
        done = False

        while not done:
            total_steps += 1
            epsilon = ddqn_agent.get_epsilon(total_steps)

            action = ddqn_agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

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
                ddqn_agent.update()


        # end_ep_time = time.time()
        # time_per_episode = end_ep_time - start_ep_time
        # total_time += end_ep_time - start_ep_time
        # time_avg_per_episode = total_time / episode_num

        # total_reward += current_ep_reward
        # episode_reward_mean = total_reward / (episode_num + 1)
        # episode_num += 1
       
        # # Save model
        # if (episode_num + 1) % save_model_episodes == 0:
        #     print(f"----- Saving model at episode {episode_num} -----")
        #     ddqn_agent.save(f"{checkpoint_base_path}_{episode_num}_{model_id}_iter.pth")    

        # # Log model info
        # if (episode_num + 1) % log_info_episodes == 0:
        #     print(f"Episode: {episode_num + 1}, Total reward: {current_ep_reward}, Mean reward: {episode_reward_mean}, Epsilon: {ddqn_agent.epsilon}")
        
        # # log movements
        # if episode_num % log_movements_episodes == 0:
        #     wandb.log({"episode_actions": wandb.Histogram(episode_actions)})

        # wandb.log({
        #     "episode_reward": current_ep_reward,
        #     "episode_reward_mean": episode_reward_mean,
        #     "epsilon": ddqn_agent.epsilon,
        #     "episode_num": episode_num,
        #     "time_per_episode": time_per_episode,
        #     "time_avg_per_episode": time_avg_per_episode
        # })    

    env.close()
    cv2.destroyAllWindows()
    # wandb.finish()


if __name__ == "__main__":
    main()