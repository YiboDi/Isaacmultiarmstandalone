import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
# from torchviz import make_dot

from copy import deepcopy

from torch.utils.data import DataLoader
import pickle

import sys
import os 
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
# sys.path.append()

from ReplayBuffer import ReplayBufferDataset
from tensorboardX import SummaryWriter
from time import time

class SAC():
    def __init__(self, 
                 network,
                 load_path = None,
                #  log_dir = None,
                #  checkpoint_dir = None,
                 experiment_dir = None,
                #  replay_buffer, 
                #  lr=3e-4,
                #  hyperparameters
                 ):
        # self.policy_net = network['policy']
        # self.q1_net = network['Q1']
        # self.q2_net = network['Q2']
        self.policy_key = 'sac_lstm'
        # self.checkpointsdir = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/checkpoints' # where are checkpoints saved
        self.network = network
        self.log_dir = experiment_dir + '/logs/Training'
        self.checkpointsdir = experiment_dir + '/checkpoints'
        self.experiment_dir = experiment_dir

        self.device = 'cuda'
        # self.writer = SummaryWriter(log_dir = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/logs/Training')
        self.writer = SummaryWriter(log_dir = self.log_dir)
        self.policy_lr = 0.0005
        self.q_lr = 0.001

        # self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        # self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.q_lr)
        # self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.q_lr)

        # capacity = 500000 #(10^4 to 10^6 )
        self.replay_buffer_capacity = 50000
        data_dic = {'observations':[],
                    'actions':[],
                    'rewards':[],
                    'next_observations':[],
                    'is_terminal':[]}
        self.replay_buffer = ReplayBufferDataset(data=data_dic, device=self.device, capacity=self.replay_buffer_capacity)
        # self.train_frequency = 100000

        # modify batch size for test
        # self.batch_size = 256
        self.batch_size = 4096 #(Typically 64-256 for SAC algorithms), 4096 from default setting
        # self.batch_size = 3

        self.num_updates_per_train = 10 #train_epoch() for 10 times each train()
        # self.tau = 0.05
        self.tau = 0.001

        self.last_train_size = 0
        # learn at start for test
        # self.warmup_steps = 10
        self.warmup_steps = 20000
        self.minimum_replay_buffer_freshness = 0.7
        self.discount = 0.99
        self.alpha = 0.001
        # self.memory_cluster_capacity = 50000
        self.reparametrize = True
        self.action_scaling = 1
        self.reward_scale = 1
        self.Q_criterion = torch.nn.MSELoss()
        # self.save_interval = 500
        # self.save_interval = 100
        self.save_interval = 10

        self.deterministic = True

        # Other hyperparameters and SAC components would be initialized here...
            # "pi_lr": 0.0005,
            # "q_lr": 0.001,
            # "discount": 0.99,
            # "tau": 0.001,
            # "alpha": 0.001,
            # "batch_size": 4096,
            # "warmup_timesteps": 20000.0,
            # "memory_cluster_capacity": 50000.0,
            # "num_updates_per_train": 10,
            # "minimum_replay_buffer_freshness": 0.7,
            # "action_scaling": 1

        self.policy = self.network['policy'].to(self.device)
        self.Q1 = self.network['Q1'].to(self.device)
        self.Q2 = self.network['Q2'].to(self.device)
        # self.policy = self.network['policy']
        # self.Q1 = self.network['Q1']
        # self.Q2 = self.network['Q2']

        self.Q1_target = deepcopy(self.Q1)
        self.Q2_target = deepcopy(self.Q2)
        self.policy_opt = optim.Adam(
            self.policy.parameters(),
            lr=self.policy_lr,
            betas=(0.9, 0.999))
        self.Q1_opt = optim.Adam(
            self.Q1.parameters(),
            lr=self.q_lr,
            betas=(0.9, 0.999))
        self.Q2_opt = optim.Adam(
            self.Q2.parameters(),
            lr=self.q_lr,
            betas=(0.9, 0.999))
        
        self.stats = {
            'time_steps': 0,
            'update_steps': 0
        }
        self.prev_update_time = time()

        if load_path is not None:
            print("[SAC] loading networks from ", load_path)
            checkpoint = torch.load(load_path, map_location=self.device)
            networks = checkpoint['networks']
            self.policy.load_state_dict(networks['policy'])
            self.Q1.load_state_dict(networks['Q1'])
            self.Q2.load_state_dict(networks['Q2'])
            self.Q1_target.load_state_dict(networks['Q1_target'])
            self.Q2_target.load_state_dict(networks['Q2_target'])
            self.policy_opt.load_state_dict(networks['policy_opt'])
            self.Q1_opt.load_state_dict(networks['Q1_opt'])
            self.Q2_opt.load_state_dict(networks['Q2_opt'])
            self.stats['time_steps'] = checkpoint['stats']['time_steps']
            self.stats['update_steps'] = checkpoint['stats']['update_steps']
            print("[SAC] Continuing from time step ", self.stats['time_steps'],
                  " and update step ", self.stats['update_steps'])

    def inference(self, observations, 
                #   deterministic=True
                  ):
        # observations = torch.FloatTensor(observations, device = self.device)
        # should spilt observations into observation here and input observation one by one into policy
        self.stats['time_steps'] += 1
        actions, action_logprobs = self.policy(observations, deterministic = self.deterministic, reparametrize = self.reparametrize)
        actions = actions.detach()
        # observations = torch.chunk(observations, observations.size(0), dim=0)
        # for i, observation in enumerate(observations):
        #     # observation = observation.squeeze(0)
        #     action, action_logprob = self.policy(observation, deterministic = self.deterministic, reparametrize = self.reparametrize)
        #     if i == 0:
        #         actions = action
        #         if self.deterministic == False:
        #             action_logprobs = action_logprob
        #     else:
        #         actions = torch.cat((actions, action), dim=0)
        #         if self.deterministic == False:
        #             action_logprobs = torch.cat((action_logprobs, action_logprob), dim=0)
        # deterministic or not has already been considered in policy_net
        # if deterministic == True:
        #     action = mean
        # else:
        #     std = log_std.exp()
        #     normal = Normal(mean, std) # distribution
        #     action = normal.sample()
        
        if self.replay_buffer is not None and len(self.replay_buffer) > self.warmup_steps and self.replay_buffer.freshness > self.minimum_replay_buffer_freshness: #change
            self.train()
            self.last_train_size = len(self.replay_buffer)
            # if self.last_train_size == self.replay_buffer_capacity:
            #     with open('replay_buffer.pkl', 'wb') as f:
            #         pickle.dump(self.replay_buffer, f)
        

        return actions

    def train(self):
        # Sample a batch from the replay buffer
        # Compute target Q values, current Q values
        # Compute policy loss, Q1 and Q2 loss
        # Update policy, Q1 and Q2 networks using optimizers
        # Optional: update target networks using polyak averaging

        # torch.autograd.set_detect_anomaly(True)

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
            # no idea why dtype of actions becomes float64, which is different from others (float32)
            # so turn all tensor in batch into float32
            for i in range(len(batch)):
                if batch[i].dtype == torch.float64:
                    batch[i] = batch[i].to(dtype = torch.float32)

            policy_loss, q1_loss, q2_loss, policy_entropy, policy_value = \
                self.train_batch(batch=batch)
            policy_loss_sum += policy_loss
            q1_loss_sum += q1_loss
            q2_loss_sum += q2_loss
            policy_entropy_sum += policy_entropy
            policy_value_sum += policy_value
            self.update_targets()
            self.stats['update_steps'] += 1
            if self.stats['update_steps'] % self.save_interval == 0:
                self.save()
                # torch.save(self.network, )


        policy_loss_sum /= self.num_updates_per_train
        q1_loss_sum /= self.num_updates_per_train
        q2_loss_sum /= self.num_updates_per_train
        policy_entropy_sum /= self.num_updates_per_train
        policy_value_sum /= self.num_updates_per_train

        # self.writer.add_scalars('01.19',{
        #     'Training/Policy_Loss': policy_loss_sum,
        #     'Training/Policy_Entropy': policy_entropy_sum,
        #     'Training/Policy_Value': policy_value_sum,
        #     'Training/Q1_Loss': q1_loss_sum,
        #     'Training/Q2_Loss': q2_loss_sum,
        #     'Training/Freshness': self.replay_buffer.freshness,
        # }, self.stats['update_steps'])
        self.writer.add_scalar('Policy_Loss', policy_loss_sum, self.stats['update_steps'])
        self.writer.add_scalar('Policy_Entropy', policy_entropy_sum, self.stats['update_steps'])
        self.writer.add_scalar('Policy_Value', policy_value_sum, self.stats['update_steps'])
        self.writer.add_scalar('Q1_Loss', q1_loss_sum, self.stats['update_steps'])
        self.writer.add_scalar('Q2_Loss', q2_loss_sum, self.stats['update_steps'])

        output = "\r[SAC] pi: {0:.4f} | pi_entropy: {1:.4f}".format(
            policy_loss_sum, policy_entropy_sum)
        output += "| pi_value: {0:.4f} ".format(policy_value_sum)
        output += "| Q1: {0:.4f} | Q2: {1:.4f} ".format(
            q1_loss_sum, q2_loss_sum)
        output += "| freshness: {0:.3f} ".format(self.replay_buffer.freshness)
        output += "| time: {0:.2f}".format(
            float(time() - self.prev_update_time))
        print(output, end='')
        self.prev_update_time = time()
        self.replay_buffer.freshness = 0.0

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
            # obs, next_obs = obs.squeeze(dim = 1), next_obs.squeeze(dim = 1)
            critic_obs = obs
            critic_next_obs = next_obs
        else:
            critic_obs, obs, actions, rewards, critic_next_obs,\
                next_obs, terminals = batch

        rewards = torch.unsqueeze(rewards, dim=1)
        terminals = torch.unsqueeze(terminals, dim=1)

        new_obs_actions, new_obs_action_logprobs = self.policy(
            obs=obs,
            deterministic=False,
            reparametrize=self.reparametrize)
        new_obs_action_logprobs = torch.unsqueeze(
            new_obs_action_logprobs, dim=1)

        new_next_obs_action, next_obs_action_logprobs = self.policy(
            obs=next_obs,
            deterministic=False,
            reparametrize=self.reparametrize)
        next_obs_action_logprobs = torch.unsqueeze(
            next_obs_action_logprobs, dim=1)

        with torch.no_grad():
            q_new_actions = torch.min(
                self.Q1(obs=critic_obs, actions=new_obs_actions *
                        self.action_scaling),
                self.Q2(obs=critic_obs, actions=new_obs_actions *
                        self.action_scaling))

        # # Train policy
        # policy_loss = (self.alpha * new_obs_action_logprobs -
        #                q_new_actions).mean()

        # self.policy_opt.zero_grad()
        # # policy_loss.backward(retain_graph=True)
        # policy_loss.backward()
        # self.policy_opt.step()

        # Train Q networks
        # with torch.autograd.set_detect_anomaly(True):
            # q1_pred = self.Q1(obs=critic_obs, actions=actions *
            #                 self.action_scaling)
            # q2_pred = self.Q2(obs=critic_obs, actions=actions *
            #                 self.action_scaling)

        target_q_values = torch.min(
            self.Q1_target(critic_next_obs, new_next_obs_action *
                        self.action_scaling),
            self.Q2_target(critic_next_obs, new_next_obs_action *
                        self.action_scaling)) \
            - (self.alpha * next_obs_action_logprobs)
        with torch.no_grad():
            q_target = self.reward_scale * rewards + (1. - terminals) * \
                self.discount * target_q_values
        # q_target = q_target.detach()
            
        q1_pred = self.Q1(obs=critic_obs, actions=actions *
                        self.action_scaling)
        # for name, param in self.Q1.named_parameters():
        #     print(f'{name}: shape={param.shape}, dtype={param.dtype}, version={param._version}')
        # dot = make_dot(q1_pred, params=dict(self.Q1.named_parameters()))
        # dot.view()
        q2_pred = self.Q2(obs=critic_obs, actions=actions *
                        self.action_scaling)

        q1_loss = self.Q_criterion(q1_pred, q_target)
        # for name, param in self.Q1.named_parameters():
        #     print(f'{name}: shape={param.shape}, dtype={param.dtype}, version={param._version}')
        q2_loss = self.Q_criterion(q2_pred, q_target)

        # test
        # critic_loss = q1_loss + q2_loss
        # self.critic_opt.zero_grad()

        self.Q1_opt.zero_grad()
        # for name, param in self.Q1.named_parameters():
        #     print(f'{name}: shape={param.shape}, dtype={param.dtype}, version={param._version}')
        # q1_pred.backward(retain_graph=True)
        # q1_loss.backward(retain_graph=True)
        q1_loss.backward()
        self.Q1_opt.step()

        self.Q2_opt.zero_grad()
        q2_loss.backward()
        self.Q2_opt.step()

        # Train policy
        policy_loss = (self.alpha * new_obs_action_logprobs -
                       q_new_actions).mean()

        self.policy_opt.zero_grad()
        # policy_loss.backward(retain_graph=True)
        policy_loss.backward()
        self.policy_opt.step()

        return (
            policy_loss.item(),
            q1_loss.item(),
            q2_loss.item(),
            -new_obs_action_logprobs.mean().item(),
            q_new_actions.detach().mean().item()
        )
    
    def save(self):
        # if self.training and self.logdir is not None:
        if self.checkpointsdir is not None:
            output_path = "{}/ckpt_{}_{:05d}".format(
                self.checkpointsdir,
                self.policy_key,
                int(self.stats['update_steps'] / self.save_interval))
            if not os.path.exists(self.checkpointsdir):
                os.makedirs(self.checkpointsdir)
            torch.save({
                'networks': self.get_state_dicts_to_save(),
                'stats': self.get_stats_to_save()
            }, output_path)
            print("[SAC] saved checkpoint at {}".format(output_path))
        # replaybufferdir = self.experiment_dir + '/replaybuffers/replay_buffer_{:05d}.pkl'.format(int(self.stats['update_steps'] / self.save_interval))
        # with open(replaybufferdir, 'wb') as f:
        #     pickle.dump(self.replay_buffer, f)
            
        rbdir = self.experiment_dir + '/replaybuffers'
        if not os.path.exists(rbdir):
            os.makedirs(rbdir)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Replay_buffer_{int(self.stats['update_steps'] / self.save_interval)}.pkl"
        replaybuffer_file = os.path.join(rbdir, filename)

        # Save the replay buffer
        with open(replaybuffer_file, 'wb') as f:
            pickle.dump(self.replay_buffer, f)

    def get_stats_to_save(self):
        return self.stats
    
    def get_state_dicts_to_save(self):
        return {
            'policy': self.policy.state_dict(),
            'Q1': self.Q1.state_dict(),
            'Q2': self.Q2.state_dict(),
            'Q1_target': self.Q1_target.state_dict(),
            'Q2_target': self.Q2_target.state_dict(),
            'policy_opt': self.policy_opt.state_dict(),
            'Q1_opt': self.Q1_opt.state_dict(),
            'Q2_opt': self.Q2_opt.state_dict(),
        }

