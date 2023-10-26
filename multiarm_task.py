# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.cloner import Cloner
import omni.kit

from gym import spaces
import numpy as np
import torch
import math
from tasks import TaskLoader
from utils import load_config

from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView



class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None) -> None:

        self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')

        self.taskloader = TaskLoader(root_dir='/home/tp2/papers/decentralized-multiarm/tasks', shuffle=True)
        self.current_task = self.taskloader.get_next_task()

        self.action_scale = 1.0
        self.dt = 1/60 # difference in time between two consecutive states or updates
        self.episode_length = self.config['environment']['episode_length']
        self.progress_buf = 0


        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def init_task(self):

        # task-specific parameters for multiagent
        

        # task-specific parameters
        self._cartpole_position = [0.0, 0.0, 2.0]
        self._reset_dist = 3.0 #reset when cart is more than 3m away from start position
        self._max_push_effort = 400.0

        #values used for defining RL buffers for multiagent and dynamic
        self._num_observation = 0 
        for item in self.config['training']['observations']['items']:
            self._num_observation += item['dimensions'] * (item['history'] + 1)

        self._num_observations = self._num_observation * self.current_task.ur5_count
        

        # values used for defining RL buffers
        # self._num_observations = 4
        self._num_action = 6 # 6 joint on ur5
        self._num_actions = self._num_action * self.current_task.ur5_count


        # self._device = "cpu"
        self._device = "gpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        # self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.obs = torch.zeros((self.num_agents, self._num_observation))
        # self.resets = torch.zeros((self.num_envs, 1))
        self.resets = torch.zeros((1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)  #[-1, +1]
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        ) #(-oo,+oo)

    def set_up_scene(self, scene) -> None:

        # eliminate all existing scene firstly
        self.scene.clear()

        self.num_agents=self.current_task.ur5_count

        self.get_franka(self.num_agents)

        self._franka_list=[]

        for i in self.num_agents:
            franka = ArticulationView(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i))
            scene.add(franka)
            self._franka_list.append(franka)
        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        # self._franka = ArticulationView(prim_paths_expr="/World/Cartpole*", name="franka_view")
        # add Cartpole ArticulationView and ground plane to the Scene
        # scene.add(self._franka)
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()

        self.init_task()

    def get_franka(self, num_agents=1):
        
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        # add the Cartpole USD to our stage
        for i in num_agents:
           position = self.current_task.base_poses[i][0]
           orientation = self.current_task.base_poses[i][0]
           create_prim(prim_path="/World/Frankas/Franka{}".format(i), prim_type="Xform", position=position, orientation=orientation, usd_path=usd_path) 

       
        # add_reference_to_stage(usd_path, "/World/Cartpole")


    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        # self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        # self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        # indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        # self.reset(indices)
        self.current_task = self.taskloader.get_next_task()
        self.set_up_scene()

        self.num_franka_dofs = self._franka_list[0].num_dof
        self.franka_dof_pos = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)
        dof_limits = self._franka_list.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[-1] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self.num_agents, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        self.franka_dof_targets = self.franka_dof_pos
        dof_vel = torch.zeros((self.num_agents, self._franka_list[0].num_dof), device=self._device)

        for i, agent in enumerate(self._franka_list):
            self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[i, :], indices=indices)
            self._franka_list[i].set_joint_positions(self.franka_dof_pos[i, :], indices=indices)
            self._franka_list[i].set_joint_velocities(dof_vel[i, :], indices=indices)



        # # randomize DOF positions
        # dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # # randomize DOF velocities
        # dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        # self._cartpoles.set_joint_velocities(dof_vel, indices=indices)
        # for i, agent in enumerate(self._franka_list):
            


        # bookkeeping
        self.resets[env_ids] = 0
        self.progress_buf = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)
        self.actions = actions.clone().to(self._device)

        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        for i, agent in enumerate(self._franka_list):
            self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[i, :])

        self.progress_buf += 1


        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self):
        # self.obs = torch.zeros((self.num_envs, self.num_agents, self._num_observation))
        # do not consider multi env temporally
        

        for i, agent in enumerate(self._franka_list):
            dof_pos = agent.get_joint_positions()
            dof_vel = agent.get_joint_velocities()

            if dof_pos.shape[0]+dof_vel.shape[0] != self._num_observation:
                raise ValueError('dim of observation does not match')
            
            self.obs[i,:] = torch.cat(dof_pos, dof_vel) # shape of self.obs is num_robots * (num_joint_pos * num_joint_vel)


        return self.obs

    def calculate_metrics(self) -> None:
        cart_pos = self.obs[:, 0]
        cart_vel = self.obs[:, 1]
        pole_angle = self.obs[:, 2]
        pole_vel = self.obs[:, 3]

        # compute reward based on angle of pole and cart velocity
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # apply a penalty if cart is too far from center
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # apply a penalty if pole is too far from upright
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)


        current_ur5_ee_residuals = self.get_ur5_eef_residuals()
        if self.prev_ur5_ee_residuals is None:
            self.prev_ur5_ee_residuals = current_ur5_ee_residuals


        collision_penalties = np.array([
            (self.collision_penalty if ur5_state['colliding'] else 0)
            for ur5_state in state['ur5s']
        ])

        if self.cooperative_individual_reach_target:
            individually_reached_target_rewards = np.array(
                [self.individually_reach_target * state['reach_count']
                    for _ in range(len(self.active_ur5s))])
        else:
            individually_reached_target_rewards = np.array([
                (self.individually_reach_target
                    if ur5_state['reached_target']
                 else 0)
                for ur5_state in state['ur5s']
            ])
        # Only give delta rewards if ee is within a radius
        # away from the target ee
        delta_position_rewards = \
            [(prev - curr) * self.delta_reward['position']
             if curr < self.delta_reward['activation_radius']
             else 0.0
             for curr, prev in zip(
                current_ur5_ee_residuals[0],
                self.prev_ur5_ee_residuals[0])]
        delta_orientation_rewards = \
            [(prev - curr) * self.delta_reward['orientation']
             if curr_pos_res < self.delta_reward['activation_radius']
             else 0.0
             for curr, prev, curr_pos_res in zip(
                current_ur5_ee_residuals[1],
                self.prev_ur5_ee_residuals[1],
                current_ur5_ee_residuals[0])]
        # proximity penalty
        proximity_penalties = np.array([
            sum([(1 - closest_points_to_other[0][8]
                  / self.proximity_penalty_distance)
                 * self.proximity_penalty
                 for closest_points_to_other in ur5.closest_points_to_others
                 if len(closest_points_to_other) > 0 and
                 closest_points_to_other[0][8] <
                 self.proximity_penalty_distance])
            for ur5 in self.active_ur5s])

        collectively_reached_targets = (
            state['reach_count'] == len(self.active_ur5s))
        collective_reached_targets_rewards = np.array(
            [(self.collectively_reach_target
                if collectively_reached_targets
              else 0)
             for _ in range(len(self.active_ur5s))])
        self.prev_ur5_ee_residuals = current_ur5_ee_residuals
        ur5_rewards_sum = \
            collision_penalties + individually_reached_target_rewards +\
            collective_reached_targets_rewards + survival_penalties + \
            proximity_penalties + \
            delta_position_rewards + delta_orientation_rewards
        if self.centralized_policy:
            return np.array([ur5_rewards_sum.sum()])
        else:
            return ur5_rewards_sum

        return reward.item()

    def is_done(self) -> None:
        # cart_pos = self.obs[:, 0]
        # pole_pos = self.obs[:, 2]

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets 

        resets = torch.where(self.progress_buf >= self.episode_length - 1, torch.ones_like(self.resets), self.resets)

        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        self.resets = resets


        return resets.item()
