# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import sys
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/exts')
import time

# from omni.isaac.kit import SimulationApp

# simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.cloner import Cloner
import omni.kit
from pxr import Usd, UsdGeom

from gym import spaces
import numpy as np
import torch
# import math
from math import pi

import sys 
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/robots')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/OmniIsaacGymEnvs/omniisaacgymenvs')
from taskloader import TaskLoader
from utils import load_config

from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim

from ur5 import UR5
from ur5_view import UR5View
from omni.isaac.core.objects import VisualCylinder



class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None) -> None:

        self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')

        self.taskloader = TaskLoader(root_dir='/home/tp2/papers/multiarm_dataset/tasks', shuffle=True)
        self.current_task = self.taskloader.get_next_task()

        # self.action_scale = 1.0
        # self.action_scale = 7.5
        # self.action_scale = 10.0
        # self.action_scale = 15.0
        # self.action_scale = 30.0
        # self.action_scale = 60.0
        self.dt = 1/60 # difference in time between two consecutive states or updates
        # self.episode_length = self.config['environment']['episode_length']
        self.progress_buf = 0

        self._cloner = GridCloner(spacing=3)
        self._cloner.define_base_env("/World/envs")
        define_prim("/World/envs/env_0")

        # self._env = env
        self._device = "cuda"
        self._num_envs = 1

        self.collision_penalty = -1
        self.delta_pos_reward = 0
        self.delta_ori_reward = 0
        self.activation_radius = 100
        self.indiv_reach_target_reward = 1
        self.coorp_reach_target_reward = 5
        self.position_tolerance = 0.04
        self.orientation_tolerance = 0.1

        self.num_franka_dofs = 6

        self.observation_space = None
        self.action_space = None

        # self._max_episode_length = 300
        self._max_episode_length = 500 # from config

        self.dof_lower_limits = torch.tensor([-2 * pi, -2 * pi, -pi, -2 * pi, -2 * pi, -2 * pi], device=self._device)
        self.dof_upper_limits = torch.tensor([2 * pi, 2 * pi, pi, 2 * pi, 2 * pi, 2 * pi], device=self._device)

        self.success = 0

        self.max_velocity = torch.tensor([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], device=self._device) # true for real ur5
        # self.max_velocity = torch.tensor([1, 1, 1, 1, 1, 1], device=self._device)
        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def init_task(self):

        # task-specific parameters for multiagent
        

        # task-specific parameters
        # self._cartpole_position = [0.0, 0.0, 2.0]
        # self._reset_dist = 3.0 #reset when cart is more than 3m away from start position
        # self._max_push_effort = 400.0

        #values used for defining RL buffers for multiagent and dynamic
        self._num_observation = 0 
        for item in self.config['training']['observations']['items']:
            self._num_observation += item['dimensions'] * (item['history'] + 1)

        self._num_observations = self._num_observation * self.current_task.ur5_count # num of obs in single env
        

        # values used for defining RL buffers

        self._num_action = 6 # 6 joint on ur5
        self._num_actions = self._num_action * self.current_task.ur5_count # num of actions in single env

        # a few class buffers to store RL-related states
        self.ob = torch.zeros((self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self.num_agents, self.num_agents, self._num_observation), device=self._device)
        # self.resets = torch.zeros((self._num_envs, 1))
        self.resets = torch.zeros((1), device=self._device)

        # set the action and observation space for RL
        # self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)  #[-1, +1]
        # self.observation_space = spaces.Box(
        #     np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        # ) #(-oo,+oo)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self._num_action))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_agents, self._num_observation))


        self.actions = torch.zeros((self.num_agents, self._num_action), device=self._device)

        self.franka_dof_targets = torch.tensor(self.current_task.start_config, device=self._device)

        # dof_limits = self._franka_list[0].get_dof_limits()
        # self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)

        # add is_terminals to check if a individual robot terminate (collide or reach its target)
        self.is_terminals = torch.zeros((self.num_agents), device=self._device)



    def set_up_scene(self, scene, replicate_physics=True) -> None:

        # # eliminate all existing scene firstly
        # if scene != None:
        #     scene.clear()
        self._stage = get_current_stage()
        self.num_agents=len(self.current_task.start_config)

        self.get_franka(self.num_agents)

        self._franka_list=[]

        # usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/assets/ur5/ur5.usd"

        for i in range(4):

            if i < self.num_agents:
                target_pos = self.current_task.target_eff_poses[i][0]
                target_ori = self.current_task.target_eff_poses[i][1]
                target_ori = target_ori[-1:] + target_ori[:-1]
                franka = UR5View(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i),
                                target_pos = target_pos,
                                target_ori = target_ori
                                )

            # add the following variable in UR5View init
                # self._franka_list[-1].target_eff_pose = self.current_task.target_eff_poses[i] # shape of num_agents*(translation3+orientation4), precisely a list of num_agents * [3.4]
                # self._franka_list[-1].goal_config = self.current_task.goal_config[i] # goal joint states, shape same with above

            elif i >= self.num_agents:
                franka = UR5View(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i),
                                )

            scene.add(franka)
            scene.add(franka.ee)
            scene.add(franka.target)

            for link in franka.link_for_contact:
                scene.add(link)

            self._franka_list.append(franka)
        scene.add_default_ground_plane()
        

        # set default camera viewport position and target
        self.set_initial_camera_params()

        self.init_task()




    def get_franka(self, num_agents):
        

        # assets_root_path = get_assets_root_path()
        # usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/assets/ur5/ur5.usd"

        for i in range(4):
            if i < self.num_agents:
                position = torch.tensor(self.current_task.base_poses[i][0])
                orientation = torch.tensor(self.current_task.base_poses[i][1])
                orientation = orientation[[3,0,1,2]]
                default_dof_pos = self.current_task.start_config[i]
                ur5 = UR5(prim_path="/World/Franka/franka{}".format(i), translation=position, orientation=orientation, usd_path=usd_path, default_dof_pos=default_dof_pos) 
                # ur5.set_anymal_properties(self._stage, ur5.prim)
                # ur5.prepare_contacts(self._stage, ur5.prim)
                # sensor_result, sensor = omni.kit.commands.execute(
                #         "IsaacSensorCreateContactSensor",
                #         path="/sensor",
                #         parent = "/World/Frankas/Franka{}".format(i),
                #         min_threshold=0,
                #         max_threshold=10000000,
                #         color= (1, 0, 0, 1),
                #         radius=0.05, # ? what for?
                #         sensor_period=self.dt,
                #         # sensor_period=-1,
                #         # offset= Gf.Vec3d(0,0,0),
                #         translation=Gf.Vec3f(0, 0, 0),
                #         visualize=True,)
                
            elif i >= self.num_agents:
                ur5 = UR5(prim_path="/World/Franka/franka{}".format(i), translation=[0, 0, -10], 
                #    orientation=orientation, 
                   usd_path=usd_path, 
                #    default_dof_pos=default_dof_pos
                   )
                # ur5.prepare_contacts(self._stage, ur5.prim)



    def set_initial_camera_params(self, camera_position=[5, 5, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def update_task(self):
        self.current_task = self.taskloader.get_next_task()
        # return current_task

    def reset(self):
        #get the next task
        # self.current_task = self.taskloader.get_next_task()
        self.update_task()
        self.success = 0
       
        # get the config of the task
        self.num_agents=len(self.current_task.start_config)

        base_poses = self.current_task.base_poses
        start_config = self.current_task.start_config
        # goal_config = self.current_task.goal_config
        target_eff_poses = self.current_task.target_eff_poses

        self._num_observations = self._num_observation * self.current_task.ur5_count
        self._num_actions = self._num_action * self.current_task.ur5_count

        self.ob = torch.zeros((self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self.num_agents, self.num_agents, self._num_observation), device=self._device)
        self.actions = torch.zeros((self.num_agents, self._num_action), device=self._device)

        self.is_terminals = torch.zeros((self.num_agents), device=self._device)

        # reset the action and observation space for RL
        # self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)  #[-1, +1]
        # self.observation_space = spaces.Box(
        #     np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        # ) #(-oo,+oo)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self._num_action))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_agents, self._num_observation))

        self.num_franka_dofs = 6 # same to self.num_action

        self.franka_dof_targets = torch.tensor(start_config, dtype=torch.float, device=self._device)
        # dof_vel = torch.zeros((self.num_agents, self._franka_list[0].num_dof), device=self._device)
        dof_vel = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)


        for i, agent in enumerate(self._franka_list):
            if i < self.num_agents:
                pos = torch.tensor(base_poses[i][0]).unsqueeze(0)
                ori = torch.tensor(base_poses[i][1]).unsqueeze(0)
                ori = ori[:,[3,0,1,2]]
                # self._franka_list[i].set_world_poses(positions = torch.tensor(base_poses[i][0]).unsqueeze(0), orientations = torch.tensor(base_poses[i][1]).unsqueeze(0), indices = torch.tensor(0))
                self._franka_list[i].set_world_poses(positions = pos, orientations = ori, 
                                                    #  indices = torch.tensor(0)
                                                     )
                # current_pos = self._franka_list[i].get_world_poses()
                # print(str(current_pos))
                self._franka_list[i].set_joint_position_targets(torch.tensor(start_config[i]))
                self._franka_list[i].set_joint_positions(torch.tensor(start_config[i]))
                self._franka_list[i].set_joint_velocities(torch.tensor(dof_vel[i]))

                # target_eff_poses = target_eff_poses[-1:] + target_eff_poses[:-1]
                position = target_eff_poses[i][0]
                orientation = target_eff_poses[i][1]
                orientation = orientation[-1:] + orientation[:-1]

                self._franka_list[i].target.set_world_pose(position = position, orientation = orientation,)

            elif i >= self.num_agents:
                self._franka_list[i].set_world_poses(positions = torch.tensor([[0,0,-10]]), 
                                                    #  indices = torch.tensor(0)
                                                     )

                self._franka_list[i].target.set_world_pose(position = [0,0,-10])

            # else:
        # self.base_pos = torch.tensor(base_poses[:][0])
        base_pos = [sublist[0] for sublist in base_poses]
        self.base_pos = torch.tensor(base_pos)


        self.progress_buf = 0

            
    #     # bookkeeping
    #     self.resets[env_ids] = 0 # self.resets = 0
        # self.progress_buf = 0

    def pre_physics_step(self, actions) -> None:
        # reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset(reset_env_ids)
        # reset = self.resets 
        # if reset == 1:
        #     self.reset()
        phys_step_start = time.time()

        # actions = torch.tensor(actions).to(self._device)
        self.actions = actions.clone().to(self._device)
        # self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        # scaled_action should be action in (-1,+1) times max_velocity divided by simulation frequency
        scaled_action = actions * self.max_velocity * self.dt * 3 # last scaler is a custom scaler to accelerate the training
        # targets = self.franka_dof_targets + self.dt * self.actions * self.action_scale # shape of self.num_agents*self.num_action, adapt self.franka_dof_targets based on its last value, making it changing smoothly
        targets = self.franka_dof_targets + scaled_action 
        self.franka_dof_targets[:] = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[i, :]) 

        self.progress_buf += 1
        phys_step_end = time.time()
        print(f"phys_step time: {phys_step_end - phys_step_start}")

        # base_pos = self._franka_list[0].get_world_poses()
        # print(str(base_pos))

        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)

    # def get_observation(self, this_franka):
    #     pos = np.array(this_franka.get_world_poses()[0]) # the base position of this_franka
    #     sorted_franka_list = sorted(self._franka_list[:self.num_agents], reverse=True, key=lambda agent: 
    #                                 np.linalg.norm(pos - np.array(agent.get_world_poses()[0]))) # get_world_poses() should return tensor with two element: position and orientation
    #     ob = torch.zeros((self.num_agents, self._num_observation), device = self._device) 
    #     for i, agent in enumerate(sorted_franka_list[0:self.num_agents]):

    #         dof_pos = agent.get_joint_positions()
    #         # set the pos of ee identical to the pos of ee_link
            
    #         ee_pos, ee_rot = agent.ee_link.get_world_poses()[0], agent.ee_link.get_world_poses()[1]

    #         # test
    #         # print('for robot{}:'.format(i))
    #         # # print('position of ee is :' + str(ee_pos) + '\n')
    #         # print('position of ee_link is :' + str(agent.ee_link.get_world_poses()) )
    #         # # print('difference value is :' + str(ee_pos - agent.ee_link.get_world_poses()[0]) + '\n')
    #         # print('position of target is :' + str(agent.target.get_world_pose()) )
    #         pos_diff = agent.ee_link.get_world_poses()[0] - agent.target.get_world_pose()[0]
    #         pos_delta = np.linalg.norm(pos_diff)
    #         ori_diff = agent.ee_link.get_world_poses()[1] - agent.target.get_world_pose()[1]
    #         ori_delta = np.linalg.norm(ori_diff)

    #         # print('position difference is :' + str(pos_delta) )
    #         # print('orientation difference is :' + str(ori_delta) )
            


    #         agent.ee.set_world_pose(position = ee_pos.squeeze(), orientation = ee_rot.squeeze())

    #         target_eff_pose = agent.target.get_world_pose()
    #         target_eff_pose = torch.tensor(np.concatenate(target_eff_pose))
    #         target_eff_pose = torch.cat([target_eff_pose, target_eff_pose]) # observation contains historical frame of 
    #         # goal_config = agent.goal_config

    #         # link position is the center of mass, for simplification using pos of links here
    #         link_position = agent.get_link_positions() # get link positions of links, 30 for ur5, why 30? reason: 10links*3xyz

    #         base_pose = agent.get_world_poses() # get the position of the base
    #         base_pose = torch.cat(base_pose, dim=-1).squeeze()

    #         if self.progress_buf == 1:
    #         # if self.ob == torch.zeros((self.num_agents, self._num_observation)): # if first step (no history yet)
    #             ob[i, 0:6] = dof_pos
    #             ob[i, 6:12] = dof_pos
    #             ob[i, 12:15] = ee_pos
    #             ob[i, 15:19] = ee_rot
    #             ob[i, 19:22] = ee_pos
    #             ob[i, 22:26] = ee_rot
    #             ob[i, 26:40] = target_eff_pose # 7*2
    #             ob[i, 40:70] = link_position
    #             ob[i, 70:100] = link_position
    #             ob[i, 100:107] = base_pose

    #         else:
    #             ob[i, 0:6] = self.ob[i, 6:12]
    #             ob[i, 6:12] = dof_pos
    #             ob[i, 12:15] = self.ob[i, 19:22]
    #             ob[i, 15:19] = self.ob[i, 22:26]
    #             ob[i, 19:22] = ee_pos
    #             ob[i, 22:26] = ee_rot
    #             ob[i, 26:40] = target_eff_pose # 7*2
    #             ob[i, 40:70] = self.ob[i, 70:100]
    #             ob[i, 70:100] = link_position
    #             ob[i, 100:107] = base_pose
    #     self.ob = ob.clone()
    #     # print('end of one step \n')

    #     return self.ob # observation of a single franka (this_franka), shape of num_agents*ob of a single agent


    # def get_observations(self):
    #     # do not consider multi env temporally
        
    #     # firstly sort the self._franka_list by base distance, furthest to closest.
    #     obs_start = time.time()

    #     obs = torch.zeros((self.num_agents, self.num_agents, self._num_observation), device = self._device)
    #     for i, agent in enumerate(self._franka_list[0:self.num_agents]):

            
    #         obs[i, :, :] = self.get_observation(this_franka=agent)


    #         # hand_pos, hand_rot = agent._hands.get_world_poses()
    #         # dof_pos = agent.get_joint_positions()
    #         # # dof_vel = agent.get_joint_velocities()


    #         # if dof_pos.shape[0]+dof_vel.shape[0] != self._num_observation:
    #         #     raise ValueError('dim of observation does not match')
            
    #         # self.obs[i,:] = torch.cat(dof_pos, hand_pos, hand_rot) # shape of self.obs is num_robots * (num_joint_pos * num_joint_vel)
    #     self.obs = obs.clone()
    #     obs_end = time.time()
    #     print('get_observations time :' + str(obs_end - obs_start) )

    #     return self.obs # observation of the whole system, shape of num_agents*num_agents*ob of a single agent(107)


    def get_observation(self):
        # ob = torch.zeros((self.num_agents, self._num_observation), device = self._device)
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            dof_pos = agent.get_joint_positions()
            ee_pos, ee_rot = agent.ee_link.get_world_poses()[0], agent.ee_link.get_world_poses()[1]
            agent.ee.set_world_pose(position = ee_pos.squeeze(), orientation = ee_rot.squeeze())
            target_eff_pose = agent.target.get_world_pose()
            target_eff_pose = torch.cat(target_eff_pose) # observation contains historical frame of
            target_eff_pose = torch.cat([target_eff_pose, target_eff_pose]) # observation contains historical frame of

            link_position = agent.get_link_positions() # get link positions of links, 30 for ur5, why 30? reason: 10links*3xyz
            base_pose = agent.get_world_poses() # get the position of the base
            base_pose = torch.cat(base_pose, dim=-1).squeeze()
            if self.progress_buf == 1:
                self.ob[i, 0:6] = dof_pos
                self.ob[i, 6:12] = dof_pos
                self.ob[i, 12:15] = ee_pos
                self.ob[i, 15:19] = ee_rot
                self.ob[i, 19:22] = ee_pos
                self.ob[i, 22:26] = ee_rot
                self.ob[i, 26:40] = target_eff_pose # 7*2
                self.ob[i, 40:70] = link_position
                self.ob[i, 70:100] = link_position
                self.ob[i, 100:107] = base_pose
            else:
                self.ob[i, 0:6] = self.ob[i, 6:12]
                self.ob[i, 6:12] = dof_pos
                self.ob[i, 12:15] = self.ob[i, 19:22]
                self.ob[i, 15:19] = self.ob[i, 22:26]
                self.ob[i, 19:22] = ee_pos
                self.ob[i, 22:26] = ee_rot
                self.ob[i, 26:40] = target_eff_pose # 7*2
                self.ob[i, 40:70] = self.ob[i, 70:100]
                self.ob[i, 70:100] = link_position
                # base_pose doesn't change, so no need to update it

        return self.ob
    
    def get_observations(self):
        start = time.time()

        self.get_observation()
        # add a new dimension at the beginning and expand along the first dimension
        self.obs = self.ob.unsqueeze(0).expand(self.num_agents,-1,-1)

        # print('before reorder:' + str(self.obs[:,:,:3]))
        distance = torch.cdist(self.base_pos, self.base_pos) # shape is [num_agent, num_agent]
        sorted_index = distance.argsort(descending = True).to(self._device) #[num_agent, num_agent]
        # expand the sorted_index to the same shape with self.obs
        expanded_index = sorted_index.unsqueeze(-1).expand(-1,-1, self._num_observation).to(self._device) # shape is [num_agent, num_agent, num_observation]

        self.obs = torch.gather(self.obs, dim=1, index=expanded_index) # shape is [num_agent, num_agent, num_observation]
        # dim=1 means gathering along the first dimension
        # self.obs[:] = self.obs[expanded_index]
        # print('after reorder:' + str(self.obs[:,:,:3]))

        end = time.time()
        print('get_observations time :' + str(end - start) )

        return self.obs

    def check_collision(self, agent):

        # self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        # # for agent in self._franka_list:
        # #     raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(agent.prim_path + "/sensor")
        # #     if raw_readings.shape[0]:                
        # #         for reading in raw_readings:
        # #             if "franka" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
        # #                 return True # Collision detected with some part of the robot
        # #             if "franka" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
        # #                 return True # Collision detected with some part of the robot

        # raw_readings = self._contact_sensor_interface.get_sensor_readings(agent.prim_path + '/sensor')
        # if raw_readings[-1] == 0:
        #     return 0
        # else:
        #     return 1
        for i, link in enumerate(agent.link_for_contact):
            if link.get_net_contact_forces() is None:
                continue
            else:
                contact = (torch.norm(link.get_net_contact_forces()) > 1.0)

                force = link.get_net_contact_forces()

                if contact:
                    print('collision happens with link at path:' + str(link.prims)) # or link.prims
                    # set the is_terminal to be 1
                    self.is_terminals[self._franka_list.index(agent)] = 1
                    return 1
            
        return 0 



    def all_reach_targets(self):
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):

            pos_delta = np.linalg.norm(agent.ee_link.get_world_poses()[0] - agent.target.get_world_pose()[0])
            ori_delat = np.linalg.norm(agent.ee_link.get_world_poses()[1] - agent.target.get_world_pose()[1])
            if pos_delta > self.position_tolerance or ori_delat > self.orientation_tolerance:
                return 0

        return 1
    
    def indiv_reach_targets(self, agent):
        pos_delta = np.linalg.norm(agent.ee_link.get_world_poses()[0] - agent.target.get_world_pose()[0])
        ori_delta = np.linalg.norm(agent.ee_link.get_world_poses()[1] - agent.target.get_world_pose()[1])
        if pos_delta < self.position_tolerance and ori_delta < self.orientation_tolerance:
            # the agent terminates if reaches its target
            self.is_terminals[self._franka_list.index(agent)] = 1
            return 1
        else:
            return 0


    def calculate_metrics(self) -> None: # calculate the rewards in each env.step()

        # collision_penalties = np.array(self.collision_penalty if self.check_collision() else 0)
        cal_start = time.time()

        collision_penalties = np.zeros(self.num_agents)
        if self.progress_buf > 1:
            for i, agent in enumerate(self._franka_list[0:self.num_agents]):
                collision = self.check_collision(agent=agent)
                if collision == 1:
                    collision_penalties[i] = self.collision_penalty # -0.05
                elif collision == 0:
                    collision_penalties[i] = 0
                else:
                    raise ValueError('The reading of the contact sensor makes no sense')


        indiv_reach_target_rewards = np.zeros(self.num_agents)
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            # if i < self.num_agents:
            if self.indiv_reach_targets(agent=agent):
                indiv_reach_target_rewards[i] = self.indiv_reach_target_reward # 0.01
            elif self.indiv_reach_targets(agent=agent) == 0:
                indiv_reach_target_rewards[i] = 0
            else:
                raise ValueError('The agent should either reach its target or not')

        pos_rewards = np.zeros(self.num_agents)
        ori_rewards = np.zeros(self.num_agents)
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            pos_delta = np.linalg.norm(agent.ee_link.get_world_poses()[0] - agent.target.get_world_pose()[0])
            ori_delta = np.linalg.norm(agent.ee_link.get_world_poses()[1] - agent.target.get_world_pose()[1])
            # if pos_delta < self.position_tolerance:
            #     pos_rewards[i] = 1
            #     if ori_delta < self.orientation_tolerance:
            #         ori_rewards[i] = 1
            #     else:
            #         ori_rewards[i] = 1.0 / (1.0 + ori_delta ** 2)
            # elif pos_delta < self.position_tolerance * 2:
            #     pos_rewards[i] = 1.0 / (1.0 + pos_delta ** 2)
            #     if ori_delta < self.orientation_tolerance:
            #         ori_rewards[i] = 1
            #     else:
            #         ori_rewards[i] = 1.0 / (1.0 + ori_delta ** 2)
            # else:
            #     pos_rewards[i] = (1.0 / (1.0 + pos_delta ** 2)) ** 2
            # Smooth, continuous reward for getting closer to the target position
            pos_rewards[i] = np.exp(-pos_delta / self.position_tolerance)

            # Smooth, continuous reward for aligning orientation to the target
            ori_rewards[i] = np.exp(-ori_delta / self.orientation_tolerance)


        
        if self.all_reach_targets():
            collectively_reach_targets_reward = np.full((self.num_agents, ), self.coorp_reach_target_reward) # 1
        else:
            collectively_reach_targets_reward = np.zeros(self.num_agents)

        franka_rewards_sum = \
            collision_penalties + indiv_reach_target_rewards +\
            collectively_reach_targets_reward \
            + pos_rewards + ori_rewards

        reward = franka_rewards_sum

        cal_end = time.time()
        print('calculate_metrics time: ', cal_end - cal_start)

        return reward

    def is_done(self):

        # resets = torch.where(self.check_collision(), torch.ones_like(self.resets), self.resets)
        # resets = torch.where(self.progress_buf >= self.episode_length - 1, torch.ones_like(self.resets), self.resets)
        resets = 0
        # resets = torch.where(self.check_collision(), 1, resets)

        # reset when collide
        for i,agent in enumerate(self._franka_list[0:self.num_agents]):
            # if i < self.num_agents:
            collision = self.check_collision(agent=agent)
            if collision == 1:
                resets = 1
                print('end episode because of collision')
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        # reset when all robots reach targets
        # for i ,agent in enumerate(self._franka_list[0:self.num_agents]):
        #     if np.linalg.norm(agent.ee_link.get_world_poses()[0] - agent.target.get_world_pose()[0]) > 0.5 * self.position_tolerance \
        #        and np.linalg.norm(agent.ee_link.get_world_poses()[1] - agent.target.get_world_poses()[1]) > 0.5 * self.orientation_tolerance:
        #         #  pass # make up
        #         resets = 1
        #         print('end episode because of success')
        if self.all_reach_targets():
            resets = 1
            self.success = 1
            print('end episode because of success')
            

        # reset when reach max steps
        # resets = 1 if self.progress_buf >= self._max_episode_length else resets
        if self.progress_buf >= self._max_episode_length:
            resets = 1
            print('end episode because of max steps')

        self.resets = resets
        if self.resets ==1:
            print('progress_buf: ', self.progress_buf)

        return resets
        # return resets.item()
