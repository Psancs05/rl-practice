import numpy as np
import random
from tinygrad import dtypes, Tensor, nn, TinyJit
from tinygrad.nn.state import safe_save, get_state_dict


# Create the buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DDQNT:
    def __init__(self, input_channels, action_dim, img_dim):

        self.layers = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), Tensor.relu,
            nn.Conv2d(32, 64, kernel_size=4, stride=2), Tensor.relu,
            nn.Conv2d(64, 64, kernel_size=3, stride=1), Tensor.relu,
            lambda x: x.flatten(1),
            nn.Linear(3136, 512), Tensor.relu,
            nn.Linear(512, action_dim), Tensor.softmax
        ]   

    def __call__(self, x):
        # Forward pass
        return x.sequential(self.layers)
        
    

class DDQNTAgent:
    def __init__(self, input_channels, action_dim, lr, gamma, epsilon, eps_decay, eps_min, replay_buffer_size, bs, sync_network_steps, img_dim):
        
        # Hyperparameters
        self.action_dim = action_dim
        self.learn_step_counter = 0
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.bs = bs
        self.sync_network_steps = sync_network_steps


        # Networks
        self.online_network = DDQNT(input_channels, action_dim, img_dim)
        self.target_network = DDQNT(input_channels, action_dim, img_dim)

        # Optimizer
        self.optimizer = nn.optim.Adam(nn.state.get_parameters(self.online_network), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        online_state_dict = nn.state.get_state_dict(self.online_network)
        target_state_dict = nn.state.get_state_dict(self.target_network)
        for key in online_state_dict:
            if key in target_state_dict:
                target_state_dict[key].assign(online_state_dict[key].detach())


    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return(np.random.randint(self.action_dim))

        with Tensor.train():
            state = Tensor(state, requires_grad=False).reshape(1, *state.shape)
            return(self.online_network(state).argmax().item())
    

    def _decay_epsilon(self):
        # Choose between epsilon decay or epsilon min
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def _sync_networks(self):
        if self.learn_step_counter % self.sync_network_steps == 0:
            online_state_dict = nn.state.get_state_dict(self.online_network)
            target_state_dict = nn.state.get_state_dict(self.target_network)
            
            for key in online_state_dict:
                if key in target_state_dict:
                    target_state_dict[key].assign(online_state_dict[key].detach())
            
        self.learn_step_counter += 1

    def update(self):
        if len(self.replay_buffer) < self.bs:
            return

        # Sample from the buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.bs)

        # Convert to tensors
        state = Tensor(state, dtype=dtypes.float32, requires_grad=False)
        action = Tensor(action, dtype=dtypes.int64, requires_grad=False)
        reward = Tensor(reward, dtype=dtypes.float32, requires_grad=False)
        next_state = Tensor(next_state, dtype=dtypes.float32, requires_grad=False)
        done = Tensor(done, dtype=dtypes.float32, requires_grad=False)

        with Tensor.train():
            # Use the online network to get the Q values
            q_value = self.online_network(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Use the target network to get the Q values
            next_q_value = self.target_network(next_state).max(1)[0]
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            # Loss
            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Sync networks
        self._sync_networks()

        # Decay epsilon
        self._decay_epsilon()

    
    def save(self, checkpoint_path):
        state_dict = get_state_dict(self.online_network)
        safe_save(state_dict, checkpoint_path)


    # def load(self, checkpoint_path):
    #     state_dict = safe_load(checkpoint_path)