import numpy as np
import gym
from gym.spaces import Box
import cv2
from nes_py.wrappers import JoypadSpace


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
        resized_obs = cv2.resize(gray_obs, (self.new_height, self.new_width), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized_obs, axis=0)
    
    def render(self, mode='human'):
        # This is the render method for the environment
        if mode == 'rgb_array':
            gray_frame = self.observation(self.env.render())
            cv2.imshow("Grayscale Super Mario", gray_frame.squeeze())
            cv2.waitKey(1)
        else:
            return super().render()


# Wrapper that skips frames and stacks them
class SkipAndStackFramesWrapper(gym.Wrapper):
    def __init__(self, env, skip=4, stack=4):
        super(SkipAndStackFramesWrapper, self).__init__(env)
        self.skip = skip
        self.stack = stack
        shp = env.observation_space.shape
        self.frames = np.zeros((stack, shp[1], shp[2]), dtype=np.uint8)
        self.observation_space = Box(low=0, high=255, shape=(stack, shp[1], shp[2]), dtype=np.uint8)
        self.frame_idx = 0

    def reset(self):
        obs, info = self.env.reset()
        obs = np.squeeze(obs, axis=0)
        for i in range(self.stack):
            self.frames[i] = obs
        self.frame_idx = 0
        return self._get_observation(), info
    
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, done, info, _ = self.env.step(action)
            obs = np.squeeze(obs, axis=0)
            self.frames[self.frame_idx % self.stack] = obs 
            self.frame_idx += 1
            total_reward += reward
            if done:
                break
        return self._get_observation(), total_reward, done, info, _

    def _get_observation(self):
        idx = self.frame_idx % self.stack
        return np.concatenate([self.frames[idx:], self.frames[:idx]], axis=0)

    def render(self, mode='human'):
        return self.env.render(mode)
    

def apply_wrappers(env, movement, img_height, img_width):
    env = JoypadSpace(env, movement)
    env = GrayScaleAndResizeWrapper(env, img_height, img_width)
    env = SkipAndStackFramesWrapper(env, skip=4)
    return env
