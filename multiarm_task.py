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

        self.config = load_config(path=/home/tp2/papers/decentralized-multiarm/configs/default.json)

        self.taskloader = TaskLoader(root_dir=/home/tp2/papers/decentralized-multiarm/tasks, shuffle=True)
        self.current_task = self.taskloader.get_next_task()


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
        self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)  #[-1, +1]
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        ) #(-oo,+oo)

    def set_up_scene(self, scene) -> None:

        # eliminate all existing scene firstly
        self.scene.remove_object()

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

    def get_franka(num_agents=1):
        
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        # add the Cartpole USD to our stage
        for i in num_agents:
           create_prim(prim_path="/World/Frankas/Franka{}".format(i), prim_type="Xform", position=self._cartpole_position, usd_path=usd_path) 

       
        # add_reference_to_stage(usd_path, "/World/Cartpole")


    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset(indices)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self):
        dof_pos = self._cartpoles.get_joint_positions()
        dof_vel = self._cartpoles.get_joint_velocities()

        # collect pole and cart joint positions and velocities for observation
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs[:, 0] = cart_pos
        self.obs[:, 1] = cart_vel
        self.obs[:, 2] = pole_pos
        self.obs[:, 3] = pole_vel

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

        return reward.item()

    def is_done(self) -> None:
        cart_pos = self.obs[:, 0]
        pole_pos = self.obs[:, 2]

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)

        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        self.resets = resets
        self.current_task = self.taskloader.get_next_task
        self.set_up_scene()

        return resets.item()
