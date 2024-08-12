import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p):
        super(DecisionTransformer, self).__init__()
        
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
        # process the images through the convolutional layers
        batch_size = states.shape[0]
        context_len = actions.shape[1]
        
        # get the embeddings
        state_embeddings = self.conv_layers(states)  # (batch_size, conv_out_size)
        state_embeddings = self.state_embedding(state_embeddings)  # (batch_size, h_dim)
        state_embeddings = state_embeddings.unsqueeze(1).repeat(1, context_len, 1)  # (batch_size, context_len, h_dim)
        
        action_embeddings = self.action_embedding(actions)  # (batch_size, context_len, h_dim)
        return_embeddings = self.return_embedding(returns_to_go)  # (batch_size, context_len, h_dim)
        
        position_embeddings = self.positional_embedding[:, :context_len, :]
        
        x = state_embeddings + action_embeddings + return_embeddings + position_embeddings
        x = self.ln(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # select the last element of the sequence
        
        action_logits = self.action_head(x)
        return action_logits
    