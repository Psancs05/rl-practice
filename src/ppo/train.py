import gym
import cv2
import os

from tqdm import trange

from gym_super_mario_bros.actions import RIGHT_ONLY

from src.wrappers import apply_wrappers
from PPO import PPO


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = RIGHT_ONLY
        

def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"

    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH)


    # ========================== Hiperparameters ==========================

    max_ep_len = 1000                       # max timesteps in one episode
    max_training_timesteps = int(1e7)       # break training loop if timeteps > max_training_timesteps

    update_timestep = max_ep_len * 5        # update policy every n timesteps
    K_epochs = 80                           # update policy for K epochs in one PPO update

    eps_clip = 0.2                          # clip parameter for PPO    
    gamma = 0.99                            # discount factor

    lr_actor = 0.0003                       # learning rate for actor network
    lr_critic = 0.001                       # learning rate for critic network

    save_model_steps = 1_000_000            # save model every n timesteps
    checkpoint_base_path = "model_checkpoints/ppo/ppo_model"

    log_info_steps = 5000                    # log model info every n timesteps

    # ========================= Enviroment ==========================
    state, _ = env.reset()
    done = False

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

    # ========================== Trainig ==========================
    time_step = 0
    episode_num = 0

    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for _ in trange(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _, __ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            
            if done:
                break

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update() 

            # save model checkpoint
            if time_step % save_model_steps == 0:
                print(f"------ Saving model checkpoint at timestep {time_step} ------")
                ppo_agent.save(f"{checkpoint_base_path}_{episode_num}.pth")

            # log model info
            if time_step % log_info_steps == 0:
                print("------ Model Info ------")
                print(f"Time step: {time_step}")
                print(f"Episode: {episode_num}")
                print(f"Current Episode Reward: {current_ep_reward}")
                print("------------------------")

        episode_num += 1

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
