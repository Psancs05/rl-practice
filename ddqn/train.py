import gym
import numpy as np
import cv2
import time
from gym.spaces import Box
import os


from tqdm import trange

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from DDQN import DDQNAgent


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
        obs = np.squeeze(obs, axis=0)  # Ensure the observation is in the correct shape (84, 84)
        for i in range(self.stack):
            self.frames[i] = obs  # Initialize all frames with the first observation
        self.frame_idx = 0
        return self._get_observation(), info
    
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, done, info, _ = self.env.step(action)
            obs = np.squeeze(obs, axis=0)  # Ensure the observation is in the correct shape (84, 84)
            self.frames[self.frame_idx % self.stack] = obs  # Update the frame in a cyclic manner
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
    

# Wrapper that skips frames
class SkipFramesWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(SkipFramesWrapper, self).__init__(env)
        self.skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, done, info, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info, _


# Wrapper that stacks frames
class StackFramesWrapper(gym.Wrapper):
    def __init__(self, env, stack=4):
        super(StackFramesWrapper, self).__init__(env)
        self.stack = stack
        shp = env.observation_space.shape
        self.frames = np.zeros((stack, shp[1], shp[2]), dtype=np.uint8)
        self.observation_space = Box(low=0, high=255, shape=(stack, shp[1], shp[2]), dtype=np.uint8)
        self.frame_idx = 0

    def reset(self):
        obs, info = self.env.reset()
        obs = np.squeeze(obs, axis=0)  # Ensure the observation is in the correct shape (84, 84)
        for i in range(self.stack):
            self.frames[i] = obs  # Initialize all frames with the first observation
        self.frame_idx = 0
        return self._get_observation(), info

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        obs = np.squeeze(obs, axis=0)  # Ensure the observation is in the correct shape (84, 84)
        self.frames[self.frame_idx % self.stack] = obs  # Update the frame in a cyclic manner
        self.frame_idx += 1
        return self._get_observation(), reward, done, info, _

    def _get_observation(self):
        idx = self.frame_idx % self.stack
        return np.concatenate([self.frames[idx:], self.frames[:idx]], axis=0)

    def render(self, mode='human'):
        return self.env.render(mode)
    


def main():
    # Define the environment
    env_name = "SuperMarioBros-1-1-v0"

    env = gym.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, MOVEMENT)
    env = GrayScaleAndResizeWrapper(env, IMG_HEIGHT, IMG_WIDTH)
    # env = SkipAndStackFramesWrapper(env, skip=4, stack=4)
    # env = SkipFramesWrapper(env, skip=4)
    env = StackFramesWrapper(env, stack=4)


    # env = gym.make(env_name, apply_api_compatibility=True)
    # env = JoypadSpace(env, MOVEMENT)
    # env = SkipFrame(env, skip=4)

    # env = GrayScaleAndResizeWrapper(env, IMG_HEIGHT, IMG_WIDTH)
    # env = ResizeObservation(env, shape=(IMG_HEIGHT, IMG_WIDTH))
    # env = GrayScaleObservation(env)

    # env = FrameStack(env, num_stack=4, lz4_compress=True)


    # ========================== Hiperparameters ==========================
    max_episodes = 5000                       # max timesteps in one episode
    
    lr = 0.00025                            # learning rate
    gamma = 0.9                             # discount factor
    epsilon = 1.0                           # exploration factor

    eps_decay = 0.99999975                     # epsilon decay
    eps_min = 0.1                           # minimum epsilon
    replay_buffer_capacity = 100_000        # replay buffer capacity
    batch_size = 32                         # batch size
    sync_network_rate = 10_000              # sync network rate

    save_model_episodes = 1_000             # save model every n episodes
    checkpoint_base_path = "model_checkpoints/ddqn/ddqn_model"

    log_info_episodes = 5                 # log model info every n episodes

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

    # ========================== Training ==========================
    episode_num = 0
    max_ep_len = 1000

    while episode_num < max_episodes:
        state, _ = env.reset()
        done = False
        current_ep_reward = 0

        for _ in trange(1, max_ep_len+1):
            # time.sleep(0.5)
            action = ddqn_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            current_ep_reward += reward

            ddqn_agent.add_to_buffer(state, action, reward, next_state, done)
            ddqn_agent.update()

            state = next_state

            if done:
                break

        episode_num += 1
        # print(f"Reward: {current_ep_reward}")

        # Save model
        if (episode_num + 1) % save_model_episodes == 0:
            print(f"----- Saving model at episode {episode_num + 1} -----")
            ddqn_agent.save(f"{checkpoint_base_path}_{episode_num + 1}_iter.pth")

        # Log model info
        if (episode_num + 1) % log_info_episodes == 0:
            print(f"Episode: {episode_num + 1}, Total reward: {current_ep_reward}, Epsilon: {ddqn_agent.epsilon}, Size of replay buffer: {len(ddqn_agent.replay_buffer)}")
        
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()