import cv2

import gym
import torch
import time
from src.wrappers import apply_wrappers
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from ppo.PPO import PPO
from ddqn.DDQN import DDQNAgent
from ddqn.ddqn_t import DDQNTAgent
from transformer.model import DTAgent


IMG_HEIGHT = 84
IMG_WIDTH = 84
MOVEMENT = SIMPLE_MOVEMENT


def main():
    env = gym.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
    env = apply_wrappers(env, MOVEMENT, IMG_HEIGHT, IMG_WIDTH, skip=4, stack=2)
    env.metadata["render_modes"] = ["human", "rgb_array"]

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape
    input_channels = state_dim[0]

    # Load the model checkpoint
    checkpoint = "model_checkpoints/ppo/ppo_model_40000_drfr5fjg.pth"
    
    if "ddqn" in checkpoint:
        print("Using DDQN Agent")
        agent = DDQNAgent(state_dim, action_dim, 0.00025, 0.99, 0.5, 5**4, 0.1, 32, 10_000, 40_000)
    elif "ppo" in checkpoint:
        print("Using PPO Agent")
        agent = PPO(input_channels, action_dim, 0.0001, 0.0, 80, 0.2, img_dim=(IMG_HEIGHT, IMG_WIDTH))
    elif "dt" in checkpoint:
        agent = DTAgent(state_dim, action_dim, 4, 128, 20, 4, 0.1, 10_000)
        context_len = 20  # Se asume que context_len es 20, ajusta según tu configuración
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        raise ValueError("Unknown model checkpoint")

    for _ in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        i = 0

        if "dt" in checkpoint:
            # Inicializar secuencias para el DT
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            actions = torch.zeros((1, context_len), dtype=torch.long).to(device)
            returns_to_go = torch.zeros((1, context_len, 1), dtype=torch.float32).to(device)
            timesteps = torch.zeros((1, context_len), dtype=torch.long).to(device)
            epissode_actions = []

        # Play a single game with render mode on
        while not done:
            if "dt" in checkpoint:
                action = agent.select_action(state, actions, returns_to_go, timesteps)
                epissode_actions.append(action)
            elif "ppo" in checkpoint:
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device) if "dt" in checkpoint else next_state
            env.render()

            total_reward += reward

            if "dt" in checkpoint:
                # Desplazar las secuencias si es necesario
                if i < context_len - 1:
                    actions[0, i] = action
                    returns_to_go[0, i, 0] = reward
                    timesteps[0, i] = i
                else:
                    # Usar clone() para evitar problemas de referencia en memoria
                    actions[0, :-1] = actions[0, 1:].clone()
                    actions[0, -1] = action
                    returns_to_go[0, :-1, 0] = returns_to_go[0, 1:, 0].clone()
                    returns_to_go[0, -1, 0] = reward
                    timesteps[0, :-1] = timesteps[0, 1:].clone()
                    timesteps[0, -1] = i

                state = next_state
            else:
                state = next_state

            i += 1

        print(f"Total reward: {total_reward}, distance: {info['x_pos']}, time: {i}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()