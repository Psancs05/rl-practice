import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
        super(ActorCritic, self).__init__()

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

        # Calculate the size of the output from conv layers
        dummy_input = torch.zeros(1, input_channels, img_dim[0], img_dim[1])
        conv_out_size = self.conv(dummy_input).view(dummy_input.size(0), -1).size(1)

        # Actor network
        self.actor_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # Forward pass through the convolutional layers
        conv_out = self.conv(state).view(state.size(0), -1)
        action_probs = self.actor_fc(conv_out)
        dist = Categorical(action_probs)

        # Sample an action from the distribution
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic_fc(conv_out)

        return action.detach(), action_logprob.detach(), state_val.detach()

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
    def __init__(self, input_channels, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, img_dim):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(input_channels, action_dim, img_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor_fc.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic_fc.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(input_channels, action_dim, img_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # Select an action from the policy
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state, copy=True)).unsqueeze(0).to(device)  # Añadir batch dimension
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        