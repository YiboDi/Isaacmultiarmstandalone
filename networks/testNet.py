import torch
import torch.nn as nn

# input tensor(4*107,)
class testpolicy(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.Sequential(
            nn.Linear(4*107, 256),
            nn.Tanh()
        )
        self.mlp = nn.Sequential(
            # nn.Linear(4*107, 256),
            # nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh()
        )
    def forward(self, obs):
        obs = self.flatten(obs)
        obs = torch.zeros(4*107)
        obs[:obs.size(0)] = obs
        obs = self.feature_extractor(obs)
        logits = self.mlp(obs)
        return logits
    
class testq(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.Sequential(
            nn.Linear(4*107, 256),
            nn.Tanh()
        )
        self.mlp = nn.Sequential(
            nn.Linear(256+6, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    def forward(self, obs, actions):
        obs = self.flatten(obs)
        obs = torch.zeros(4*107)
        obs[:obs.size(0)] = obs
        obs = self.feature_extractor(obs)
        actions = actions.squeeze(0)
        input = torch.cat((obs, actions), dim=0)
        logits = self.mlp(input)
        return logits