import numpy as np
import torch
import torch.nn as nn
import random


################################## set device ##################################
print("==================================================================")
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("==================================================================")


############################## DDQN ##############################

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
    

# Deep Q-Network
class DDQN(nn.Module):
    def __init__(self, input_channels, action_dim, img_dim):
        super(DDQN, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Linear layers
        conv_out_size = self._get_conv_out((input_channels, *img_dim))
        self.network = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Forward pass
        return self.network(x)
        
    def _get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape)).to(device)
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(out.size()))

# DDQN Agent
class DDQNAgent:
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
        self.online_network = DDQN(input_channels, action_dim, img_dim).to(device)
        self.target_network = DDQN(input_channels, action_dim, img_dim).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)


    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return(np.random.randint(self.action_dim))
        
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            return(self.online_network(state).argmax().item())
    

    def _decay_epsilon(self):
        # Choose between epsilon decay or epsilon min
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


    def _sync_networks(self):
        # Sync the target network with the online network
        if self.learn_step_counter % self.sync_network_steps == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        self.learn_step_counter += 1


    def update(self):
        # Check if the buffer is full
        if len(self.replay_buffer) < self.bs:
            return

        # Sample from the buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.bs)

        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        # Q-values
        q_values = self.online_network(state)
        next_q_values = self.target_network(next_state)  
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Loss
        loss = self.loss(q_value, expected_q_value)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync networks
        self._sync_networks()

        # Decay epsilon
        self._decay_epsilon()

    
    def save(self, checkpoint_path):
        torch.save(self.online_network.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.online_network.load_state_dict(torch.load(checkpoint_path))
        self.target_network.load_state_dict(torch.load(checkpoint_path))