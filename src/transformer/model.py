import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np


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


############################## Decision Transformer ##############################

# Replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, context_len):
        self.context_len = context_len
        self.buffer_size = buffer_size
        
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]

    def add(self, state_seq, action_seq, reward_seq, next_state_seq, done_seq):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state_seq, action_seq, reward_seq, next_state_seq, done_seq))

    def clear(self):
        del self.buffer[:]


# Model
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p):
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_len = context_len
        
        # define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # get the output size of the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, 84, 84)
            conv_out_size = self.conv_layers(dummy_input).shape[1]
        
        # define embeddings
        self.state_embedding = nn.Linear(conv_out_size, h_dim)  # State embedding
        self.action_embedding = nn.Embedding(act_dim, h_dim)    # Action embedding
        self.return_embedding = nn.Linear(1, h_dim)             # Return embedding
        
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_len, h_dim))
        
        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_heads, dim_feedforward=h_dim*4, dropout=drop_p)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_blocks)
        
        # Action head
        self.action_head = nn.Linear(h_dim, act_dim)
        self.ln = nn.LayerNorm(h_dim)

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len, _, _, _ = states.shape
        
        # Process the images through the convolutional layers and flatten
        states = states.view(-1, *states.shape[-3:])
        state_embeddings = self.conv_layers(states)
        state_embeddings = self.state_embedding(state_embeddings)
        state_embeddings = state_embeddings.view(batch_size, seq_len, -1)
        
        # Process actions and returns_to_go
        action_embeddings = self.action_embedding(actions)
        
        # Expand returns_to_go if necessary
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        
        return_embeddings = self.return_embedding(returns_to_go)
        
        # Positional encodings
        timestep_embeddings = self.positional_embedding[:, :seq_len, :]
        timestep_embeddings = timestep_embeddings.expand(batch_size, -1, -1)
        
        # Add positional embeddings
        state_embeddings += timestep_embeddings
        action_embeddings += timestep_embeddings
        return_embeddings += timestep_embeddings
        
        # Concatenate all embeddings
        x = torch.cat((state_embeddings, action_embeddings, return_embeddings), dim=1)
        
        # Transformer
        x = self.ln(x)
        x = self.transformer(x)
        
        # Output
        x = x[:, -1, :]
        action_logits = self.action_head(x)
        
        return action_logits


# DT Agent
class DTAgent:
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, buffer_size):
        self.model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=h_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=drop_p
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.replay_buffer = ReplayBuffer(buffer_size, 20)
        self.context_len = context_len

    def select_action(self, states, actions, returns_to_go, timesteps):
        with torch.no_grad():
            action_logits = self.model(states, actions, returns_to_go, timesteps)
            action_prob = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_prob, num_samples=1)
        return action.item()

    
    def store_transition(self, state, action, reward, next_state, done):
        if len(self.replay_buffer.states) >= self.context_len:
            self.replay_buffer.states.pop(0)
            self.replay_buffer.actions.pop(0)
            self.replay_buffer.rewards.pop(0)
            self.replay_buffer.next_states.pop(0)
            self.replay_buffer.dones.pop(0)
        
        self.replay_buffer.states.append(state)
        self.replay_buffer.actions.append(action)
        self.replay_buffer.rewards.append(reward)
        self.replay_buffer.next_states.append(next_state)
        self.replay_buffer.dones.append(done)
        
        # Add the sequence to the replay buffer
        if len(self.replay_buffer.states) == self.context_len:
            self.replay_buffer.add(
                torch.stack(list(self.replay_buffer.states)),
                torch.tensor(list(self.replay_buffer.actions)),
                torch.tensor(list(self.replay_buffer.rewards)),
                torch.stack(list(self.replay_buffer.next_states)),
                torch.tensor(list(self.replay_buffer.dones))
            )


    def update(self, batch_size, gamma):
        if len(self.replay_buffer) >= batch_size:
            batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            
            # Get the batch data
            batch_data = [self.replay_buffer[i] for i in batch]
            states, actions, rewards, next_states, dones = zip(*batch_data)

            # Convert to tensors
            states = torch.stack(states).squeeze(2).to(device)
            actions = torch.stack([a.clone().detach().long() for a in actions]).to(device)
            rewards = torch.stack([r.clone().detach().float() for r in rewards]).unsqueeze(-1).to(device)
            next_states = torch.stack(next_states).squeeze(2).to(device)
            dones = torch.stack([d.clone().detach().float() for d in dones]).unsqueeze(-1).to(device)

            # Compute the returns-to-go
            returns_to_go = torch.zeros_like(rewards).to(device)
            future_return = 0.0
            for i in reversed(range(rewards.size(1))):
                future_return = rewards[:, i] + gamma * future_return * (1 - dones[:, i])
                returns_to_go[:, i] = future_return

            # Get the timesteps
            timesteps = torch.arange(0, self.context_len).unsqueeze(0).repeat(batch_size, 1).to(device)

            # Forward pass
            action_preds = self.model(states, actions, returns_to_go, timesteps)
            
            # Compute the loss
            loss = F.cross_entropy(action_preds, actions[:, -1])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()


    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
