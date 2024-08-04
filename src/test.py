import cv2

import gym
from src.wrappers import apply_wrappers
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from ppo.PPO import PPO
from ddqn.DDQN import DDQNAgent


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = SIMPLE_MOVEMENT


def main():
    env = gym.make('SuperMarioBrosRandomStages-v0', apply_api_compatibility=True, render_mode='human')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH, 4, 4)
    env.metadata["render_modes"] = ["human", "rgb_array"]

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]

    # Load the model checkpoint
    checkpoint = "model_checkpoints/ppo/ppo_model_695_mzu2ftkg.pth"
    
    if "ddqn" in checkpoint:
        agent = DDQNAgent(action_dim=action_dim, lr=0.00025, gamma=0.6, epsilon=0.01, eps_decay=0.999, eps_min=0.1, replay_buffer_size=100000, bs=32, sync_network_steps=10000, img_dim=(IMG_HEIGHT, IMG_WIDTH), input_channels=input_channels)
    else:
        agent = PPO(input_channels, action_dim, 0.0003, 0.001, 0.99, 80, 0.2, img_dim=(IMG_HEIGHT, IMG_WIDTH))
    

    for _ in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        i = 0

        # Play a single game with render mode on
        while not done and i < 700:
            # action = env.action_space.sample()
            action = agent.select_action(state)
            state, reward, done, _, __ = env.step(action)
            env.render()
            i+=1
            total_reward += reward

        print("Reward: ", total_reward)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()