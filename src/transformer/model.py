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


# Replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]


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
        # Process the images through the convolutional layers
        batch_size = states.shape[0]
        context_len = actions.shape[1]
        
        # Obtain the positional embeddings
        position_embeddings = self.positional_embedding[:, :context_len, :]  # (1, context_len, h_dim)
        
        # Obtain the embeddings of the states and add the positional embeddings to them
        state_embeddings = self.conv_layers(states)  # (batch_size, conv_out_size)
        state_embeddings = self.state_embedding(state_embeddings)  # (batch_size, h_dim)
        state_embeddings = state_embeddings.unsqueeze(1).repeat(1, context_len, 1)  # (batch_size, context_len, h_dim)
        state_embeddings = state_embeddings + position_embeddings  # Correct broadcasting

        # Obtain the embeddings of the actions and add the positional embeddings to them
        action_embeddings = self.action_embedding(actions)  # (batch_size, context_len, h_dim)
        action_embeddings = action_embeddings + position_embeddings  # Correct broadcasting

        # Obtain the embeddings of the returns and add the positional embeddings to them
        return_embeddings = self.return_embedding(returns_to_go)  # (batch_size, context_len, h_dim)
        return_embeddings = return_embeddings.repeat(batch_size, 1, 1) + position_embeddings  # Shape adjustment and broadcasting
        
        # Now all the embeddings have the shape (batch_size, context_len, h_dim)
        
        # Stack the embeddings along the new dimension
        x = torch.cat((state_embeddings, action_embeddings, return_embeddings), dim=1)  # (batch_size, 3*context_len, h_dim)
        
        # Apply LayerNorm and Transformer
        x = self.ln(x)
        x = self.transformer(x)
        
        # Select the last element of the sequence
        x = x[:, -1, :]  # (batch_size, h_dim)
        
        # Obtain the action logits
        action_logits = self.action_head(x)  # (batch_size, act_dim)
        
        return action_logits



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
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        

    def select_action(self, state, actions, returns_to_go, timesteps):
        with torch.no_grad():
            action_logits = self.model(state, actions, returns_to_go, timesteps)
            action_prob = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_prob, num_samples=1)
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.states.append(state)
        self.replay_buffer.actions.append(action)
        self.replay_buffer.rewards.append(reward)
        self.replay_buffer.next_states.append(next_state)
        self.replay_buffer.dones.append(done)
    
    def update(self, batch_size, gamma, timesteps):
        # train the model
        if len(self.replay_buffer) >= batch_size:
            batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            
            # sample the batch
            states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])

            states = torch.cat(states).to(device)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            next_states = torch.cat(next_states).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)

            returns_to_go = torch.zeros_like(rewards).to(device)
            future_return = 0.0
            for i in reversed(range(rewards.size(0))):
                future_return = rewards[i] + gamma * future_return * (1 - dones[i])
                returns_to_go[i] = future_return
            
            # forward pass
            action_preds = self.model(states, actions, returns_to_go, timesteps)
            
            # compute the loss
            loss = F.cross_entropy(action_preds, actions.squeeze(-1))
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # clear the replay buffer
            self.replay_buffer.clear()

            return loss.item()
    

    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)