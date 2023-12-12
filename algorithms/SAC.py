import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from torch.utils.data import DataLoader

import sys
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
sys.path.append()

from ReplayBuffer import ReplayBufferDataset
from tensorboardX import SummaryWriter
import time

class SAC(BaseRLAlgo):
    def __init__(self, 
                 network,
                #  replay_buffer, 
                 lr=3e-4):
        self.policy_net = network['policy_net']
        self.q1_net = network['q1_net']
        self.q2_net = network['q2_net']

        self.device = 'cuda'
        self.writer = SummaryWriter()

        self.policy_lr = lr
        self.q_lr = lr

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.q_lr)
        data_dic = {'observations':[],
                    'actions':[],
                    'rewards':[],
                    'next_observations':[]}
        self.replay_buffer = ReplayBufferDataset(data=data_dic, device=self.device, capacity=200000)
        self.train_frequency = 100000

        self.batch_size = 256
        self.num_updates_per_train = 10 #train_epoch() for 10 times each train()
        self.tau = 0.05

        self.last_train_size = 0
        self.warmup_steps = 100000

        # Other hyperparameters and SAC components would be initialized here...

    def inference(self, observation, deterministic=True):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        mean, log_std = self.policy_net(observation)
        std = log_std.exp()
        if deterministic == True:
            action = mean
        else:
            normal = Normal(mean, std)
            action = normal.sample()
        
        if self.replay_buffer is not None and len(self.replay_buffer) > self.warmup_steps and len(self.replay_buffer) - self.last_train_size == self.train_frequency: #change
            self.train()
            self.last_train_size = len(self.replay_buffer)

        return action

    def train(self):
        # Sample a batch from the replay buffer
        # Compute target Q values, current Q values
        # Compute policy loss, Q1 and Q2 loss
        # Update policy, Q1 and Q2 networks using optimizers
        # Optional: update target networks using polyak averaging
        policy_loss_sum = 0.0
        q1_loss_sum = 0.0
        q2_loss_sum = 0.0
        policy_entropy_sum = 0.0
        policy_value_sum = 0.0

        self.train_loader = DataLoader( # load mini-batch from replay buffer for train_batch
                    self.replay_buffer,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0)
        train_loader_it = iter(self.train_loader)
        for i in range(self.num_updates_per_train): # 10 times train_batch per train
            batch = next(train_loader_it, None)
            if batch is None:
                train_loader_it = iter(self.train_loader)
                batch = next(train_loader_it, None)
            policy_loss, q1_loss, q2_loss, policy_entropy, policy_value = \
                self.train_batch(batch=batch)
            policy_loss_sum += policy_loss
            q1_loss_sum += q1_loss
            q2_loss_sum += q2_loss
            policy_entropy_sum += policy_entropy
            policy_value_sum += policy_value
            self.update_targets()
            # self.stats['update_steps'] += 1
            # if self.stats['update_steps'] % self.save_interval == 0:
            #     self.save()


        policy_loss_sum /= self.num_updates_per_train
        q1_loss_sum /= self.num_updates_per_train
        q2_loss_sum /= self.num_updates_per_train
        policy_entropy_sum /= self.num_updates_per_train
        policy_value_sum /= self.num_updates_per_train

        self.writer.add_scalars({
            'Training/Policy_Loss': policy_loss_sum,
            'Training/Policy_Entropy': policy_entropy_sum,
            'Training/Policy_Value': policy_value_sum,
            'Training/Q1_Loss': q1_loss_sum,
            'Training/Q2_Loss': q2_loss_sum,
            'Training/Freshness': self.replay_buffer.freshness,
        })

        output = "\r[SAC] pi: {0:.4f} | pi_entropy: {1:.4f}".format(
            policy_loss_sum, policy_entropy_sum)
        output += "| pi_value: {0:.4f} ".format(policy_value_sum)
        output += "| Q1: {0:.4f} | Q2: {1:.4f} ".format(
            q1_loss_sum, q2_loss_sum)
        # output += "| freshness: {0:.3f} ".format(self.replay_buffer.freshness)
        output += "| time: {0:.2f}".format(
            float(time() - self.prev_update_time))
        print(output, end='')
        self.prev_update_time = time()
        # self.replay_buffer.freshness = 0.0

    def update_targets(self):
        for Qf_target, Qf in zip(
            [self.Q1_target, self.Q2_target],
                [self.Q1, self.Q2]):
            for target_param, param in zip(
                    Qf_target.parameters(), Qf.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) +
                                        param.data * self.tau)

    def train_batch(self, batch):
        if len(batch) == 5:
            obs, actions, rewards, next_obs, terminals = batch
            critic_obs = obs
            critic_next_obs = next_obs
        else:
            critic_obs, obs, actions, rewards, critic_next_obs,\
                next_obs, terminals = batch

        rewards = torch.unsqueeze(rewards, dim=1)
        terminals = torch.unsqueeze(terminals, dim=1)

        new_obs_actions, new_obs_action_logprobs = self.policy_net(
            obs=obs,
            deterministic=False,
            reparametrize=self.reparametrize)
        new_obs_action_logprobs = torch.unsqueeze(
            new_obs_action_logprobs, dim=1)

        new_next_obs_action, next_obs_action_logprobs = self.policy_net(
            obs=next_obs,
            deterministic=False,
            reparametrize=self.reparametrize)
        next_obs_action_logprobs = torch.unsqueeze(
            next_obs_action_logprobs, dim=1)

        q_new_actions = torch.min(
            self.q1_net(obs=critic_obs, actions=new_obs_actions *
                    self.action_scaling),
            self.q2_net(obs=critic_obs, actions=new_obs_actions *
                    self.action_scaling))

        # Train policy
        policy_loss = (self.alpha * new_obs_action_logprobs -
                       q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Train Q networks
        q1_pred = self.q1_net(obs=critic_obs, actions=actions *
                          self.action_scaling)
        q2_pred = self.q2_net(obs=critic_obs, actions=actions *
                          self.action_scaling)

        target_q_values = torch.min(
            self.Q1_target(critic_next_obs, new_next_obs_action *
                           self.action_scaling),
            self.Q2_target(critic_next_obs, new_next_obs_action *
                           self.action_scaling)) \
            - (self.alpha * next_obs_action_logprobs)
        with torch.no_grad():
            q_target = self.reward_scale * rewards + (1. - terminals) * \
                self.discount * target_q_values

        q1_loss = self.Q_criterion(q1_pred, q_target)
        q2_loss = self.Q_criterion(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        return (
            policy_loss.item(),
            q1_loss.item(),
            q2_loss.item(),
            -new_obs_action_logprobs.mean().item(),
            q_new_actions.detach().mean().item()
        )

