import gym
import numpy as np
import cv2
from gym.spaces import Box

from tqdm import trange

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from PPO import PPO


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = RIGHT_ONLY


# Crates a wrapper that converts the RGB image to grayscale and resizes it
class GrayScaleAndResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, new_width, new_height):
        super(GrayScaleAndResizeWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, new_height, new_width), dtype=np.uint8)
        self.new_width = new_width
        self.new_height = new_height
        
    def observation(self, obs):
        # Greyscale
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Resize
        resized_obs = cv2.resize(gray_obs, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized_obs, axis=0)
    
    def render(self, mode='human'):
        # This is the render method for the environment
        if mode == 'rgb_array':
            gray_frame = self.observation(self.env.render())
            cv2.imshow("Grayscale Super Mario", gray_frame.squeeze())
            cv2.waitKey(1)
        else:
            return super().render()
        


    

def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"

    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, MOVEMENT)
    env = GrayScaleAndResizeWrapper(env, IMG_HEIGHT, IMG_WIDTH)


    # ========================== Hiperparameters ==========================

    max_ep_len = 1000                       # max timesteps in one episode
    max_training_timesteps = int(100)       # break training loop if timeteps > max_training_timesteps

    update_timestep = max_ep_len * 4        # update policy every n timesteps
    K_epochs = 80                           # update policy for K epochs in one PPO update

    eps_clip = 0.2                          # clip parameter for PPO    
    gamma = 0.99                            # discount factor

    lr_actor = 0.0003                       # learning rate for actor network
    lr_critic = 0.001                       # learning rate for critic network

    # ========================= Enviroment ==========================
    state, _ = env.reset()
    done = False

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]

    print(f"Action Dimension: {action_dim}")
    print(f"State Dimension: {state_dim}")
    print(f"Input Channels: {input_channels}")


    # ========================== Model ==========================
    ppo_agent = PPO(input_channels, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, img_dim=(IMG_HEIGHT, IMG_WIDTH))


    # ========================== Trainig ==========================

    time_step = 0

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

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update() 

            # break; if the episode is over
            if done:
                break

        print("Reward: ", current_ep_reward)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    main()
    # play_game()