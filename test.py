import cv2
import numpy as np
import time

import gym
from gym.spaces import Box
from gym import Wrapper
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from ppo.PPO import PPO
from ddqn.DDQN import DDQNAgent


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
        
# Creates a wrapper taht skips frames
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


def main():
    env_name = "SuperMarioBros-1-1-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, MOVEMENT)
    env = SkipFrame(env, skip=4)

    # env = GrayScaleAndResizeWrapper(env, IMG_HEIGHT, IMG_WIDTH)
    env = ResizeObservation(env, shape=(IMG_HEIGHT, IMG_WIDTH))
    env = GrayScaleObservation(env)

    env = FrameStack(env, num_stack=4, lz4_compress=True)

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]

    # Load the model checkpoint
    checkpoint = "model_checkpoints/ddqn/ddqn_model_500_iter.pth"
    
    if "ddqn" in checkpoint:
        agent = DDQNAgent(action_dim=action_dim, lr=0.00025, gamma=0.6, epsilon=0.01, eps_decay=0.999, eps_min=0.1, replay_buffer_size=100000, bs=32, sync_network_steps=10000, img_dim=(IMG_HEIGHT, IMG_WIDTH), input_channels=input_channels)
    else:
        agent = PPO(input_dims=(4, IMG_HEIGHT, IMG_WIDTH), num_actions=action_dim)

    state, _ = env.reset()
    done = False
    total_reward = 0

    # Play a single game with render mode on
    while not done:
        # action = env.action_space.sample()
        action = agent.select_action(state)
        state, reward, done, _, __ = env.step(action)
        env.render()
        total_reward += reward

    print("Reward: ", total_reward)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()