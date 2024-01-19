import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# input tensor(4*107,)
class testpolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
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
    def forward(self, obs, deterministic, reparametrize, return_dist = False):
        actions = None
        action_logprobs = None
        obs = self.flatten(obs)
        # obs = torch.zeros(4, 4*107)
        # obs[:obs.size(0)] = obs
        obs = obs.to('cuda')
        obs = self.feature_extractor(obs)     
        output = self.mlp(obs) # dim of output is 1, size is 12

        action_means, action_variances = torch.split(
            output, 6, dim=1
        )
        action_variances = (0.1 - 1e-08) * \
            (action_variances + 1.0) / 2.0 + 1e-08
        action_variances = action_variances.exp()

        if deterministic:
            actions = action_means
        else:
            dist = self.generate_dist(action_means, action_variances)
            if return_dist:
                return dist
            if reparametrize: #true
                actions = dist.rsample()
            else:
                actions = dist.sample()
            action_logprobs = dist.log_prob(actions)

        return actions, action_logprobs
        
    def generate_dist(self, means, variances):
        variances = torch.stack(
            [torch.diag(variance)
             for variance in variances])
        try:
            dist = MultivariateNormal(means, variances)
            return dist
        except Exception as e:
            print(e)
            print("mean:\n{}\nvariances:\n{}".format(
                means, variances))
    
class testq(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
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
        # obs = torch.zeros(4*107)
        # obs[:obs.size(0)] = obs
        obs = obs.to('cuda')
        obs = self.feature_extractor(obs)
        # actions = actions.squeeze(0)
        input = torch.cat((obs, actions), dim=1)
        logits = self.mlp(input)
        return logits