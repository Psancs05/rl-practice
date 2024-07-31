import cv2
import numpy as np
import time

import gym
from gym.spaces import Box
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
    env_name = "SuperMarioBros-1-1-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, MOVEMENT)
    env = GrayScaleAndResizeWrapper(env, IMG_HEIGHT, IMG_WIDTH)


    # Load the PPO agent
    ppo_checkpoint = "model_checkpoints/ppo/ppo_model_1000000.pth"
    # ppo = PPO.load(ppo_checkpoint)

    state, _ = env.reset()
    done = False
    total_reward = 0

    # Play a single game with render mode on
    while not done:
        action = env.action_space.sample()
        state, reward, done, _, __ = env.step(action)
        env.render(mode='human')
        total_reward += reward

    print("Reward: ", total_reward)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()