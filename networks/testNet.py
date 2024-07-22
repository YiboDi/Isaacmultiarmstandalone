import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Normal

class testpolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, policy_probs=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction='sum'):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_probs = policy_probs # whether policy network generates also action variances
        self._clip_log_std = clip_log_std
        if clip_log_std:
            self._log_std_min = min_log_std
            self._log_std_max = max_log_std
        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = torch.mean if reduction == "mean" else torch.sum if reduction == "sum" \
            else torch.prod if reduction == "prod" else None
        self.flatten = nn.Flatten(start_dim=1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh()
        )

        if self.policy_probs:
            self.output_layer = nn.Linear(64, self.action_dim*2)
        elif not self.policy_probs:
            self.output_layer = nn.Linear(64, self.action_dim)

        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            self.output_layer,
        )


    def forward(self, obs, deterministic=False, reparametrize=True, return_dist = False):
        actions = None
        action_logprobs = None
        obs = self.flatten(obs)
        # obs = torch.zeros(4, 4*107)
        # obs[:obs.size(0)] = obs
        obs = obs.to('cuda')
        obs = self.feature_extractor(obs)     
        output = self.mlp(obs) # dim of output is 1, size is 12

        if self.policy_probs:
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


        elif not self.policy_probs:
            action_means = output
            action_variances = nn.Parameter(torch.zeros(self.action_dim, device='cuda'))
            # clamp log standard deviations
            if self._clip_log_std:
                action_variances = torch.clamp(action_variances, self._log_std_min, self._log_std_max)

            if deterministic:
                actions = action_means
            else:
                # distribution
                self._distribution = Normal(action_means, action_variances.exp())

                # sample using the reparameterization trick
                actions = self._distribution.rsample()

                # log of the probability density function
                action_logprobs = self._distribution.log_prob(actions)
                if self._reduction is not None:
                    action_logprobs = self._reduction(action_logprobs, dim=-1)
                if action_logprobs.dim() != actions.dim():
                    action_logprobs = action_logprobs.unsqueeze(-1)
                    # action_logprobs = torch.stack([action_logprobs]*self.action_dim, dim=-1)

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
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh()
        )
        self.mlp = nn.Sequential(
            nn.Linear(256+action_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            # nn.Tanh()
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