import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import trange

################################## set device ##################################
print("==================================================================")
device = torch.device('cpu')
if torch.cuda.is_available(): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device set to : " + str(device))
else:
    print("Device set to : cpu")
print("==================================================================")


############################## PPO  ##############################

# Create the buffer
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim, img_dim):
        super().__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        
        # # Calculate the size of the output from conv layers
        # dummy_input = torch.zeros(1, input_channels, img_dim[0], img_dim[1])
        # conv_out_size = self.conv(dummy_input).view(dummy_input.size(0), -1).size(1)
        # print("Convolutional output size: ", conv_out_size); exit()

        # Actor network
        self.actor_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        action_probs = self.actor_fc(conv_out)
        state_val = self.critic_fc(conv_out)
        return action_probs, state_val

    def evaluate(self, old_states, old_actions):
        # Evaluate the old actions and values from the buffer
        if old_states.dim() == 3:  # Check if tensor has shape (batch_size, height, width)
            old_states = old_states.unsqueeze(1) 
            
        conv_out = self.conv(old_states).view(old_states.size(0), -1)
        action_probs = self.actor_fc(conv_out)
        
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(old_actions)
        state_values = self.critic_fc(conv_out).squeeze()
        dist_entropy = dist.entropy().mean()
        
        return action_logprobs, state_values, dist_entropy



# PPO algorithm
class PPO:
    def __init__(self, input_channels, action_dim, lr, gamma, K_epochs, eps_clip, img_dim):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(input_channels, action_dim, img_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(input_channels, action_dim, img_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state, copy=True)).unsqueeze(0).to(device)
            action_probs, state_val = self.policy(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob
            

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path))
        self.policy_old.load_state_dict(self.policy.state_dict())


    def store_transition(self, state, action, action_logprob, reward, done):
        self.buffer.states.append(torch.tensor(state, dtype=torch.float32))
        self.buffer.actions.append(torch.tensor(action))
        self.buffer.logprobs.append(action_logprob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def compute_advantages(self, rewards, values, dones, gamma, gae_lambda=0.95):
        advantages = []
        gae = 0
        values = torch.cat((values, torch.tensor([0.0]).to(device)), dim=0)

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        return advantages
    

    def update(self):
        states = torch.stack(self.buffer.states).to(device).detach()
        actions = torch.stack(self.buffer.actions).to(device).detach()
        logprobs = torch.stack(self.buffer.logprobs).to(device).detach()
        rewards = self.buffer.rewards
        dones = self.buffer.is_terminals

        # Compute the state values using the critic network
        _, values = self.policy(states)
        values = values.squeeze().detach()

        # Compute the advantages
        advantages = self.compute_advantages(rewards, values, dones, self.gamma)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device).detach()

        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluate actions and calculate new log_probabilities
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)
            entropy_loss = dist.entropy().mean()

            # Calculate the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - logprobs.detach())

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values.squeeze(), torch.tensor(rewards, dtype=torch.float32).to(device)) - 0.01 * entropy_loss
            # Optimize the policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear the buffer
        self.buffer.clear()

        return loss.item()
