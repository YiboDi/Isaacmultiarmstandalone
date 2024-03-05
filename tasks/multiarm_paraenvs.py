import sys
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/exts')

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
from omni.isaac.core.prims import XFormPrimView, GeometryPrimView

from gym import spaces
import numpy as np
import torch
# import math
from math import pi

import sys 
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/robots')
from taskloader import TaskLoader
from utils import load_config

from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim

# from ur5withEEandTarget import UR5withEEandTarget
from ur5 import UR5
from ur5_view import UR5View
from ur5_view_multienv import UR5MultiarmEnv
from omni.isaac.core.objects import VisualCylinder



class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None, env=None) -> None:
        self.mode = 'supervision'
        self._env = env

        self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')

        self.taskloader = TaskLoader(root_dir='/home/tp2/papers/multiarm_dataset/tasks', shuffle=True)
        # self.current_task = self.taskloader.get_next_task()
        self._num_envs = 1
        self._env_spacing = 3
        # self.current_tasks = []
        # for i in range(self._num_envs):
        #     current_task = self.taskloader.get_next_task()
        #     while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
        #         current_task = self.taskloader.get_next_task()

        #     self.current_tasks.append(self.taskloader.get_next_task())

        # self.num_agents=len(self.current_tasks[0].start_config)

        self.dt = 1/60 # difference in time between two consecutive states or updates

        self.progress_buf = 0

        self.default_zero_env_path = '/World/envs/env_0'
        self.default_base_env_path = '/World/envs'

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self._device = "cuda"

        self.collision_penalty = -1
        self.delta_pos_reward = 0
        self.delta_ori_reward = 0
        self.activation_radius = 100
        self.indiv_reach_target_reward = 1
        self.coorp_reach_target_reward = 5
        self.position_tolerance = 0.04
        self.orientation_tolerance = 0.1

        self.num_franka_dofs = 6

        self._max_episode_length = 300
        # self._max_episode_length = 500 # from config

        self.dof_lower_limits = torch.tensor([-2 * pi, -2 * pi, -pi, -2 * pi, -2 * pi, -2 * pi], device=self._device)
        self.dof_upper_limits = torch.tensor([2 * pi, 2 * pi, pi, 2 * pi, 2 * pi, 2 * pi], device=self._device)

        self.success = torch.zeros((self._num_envs), device=self._device)

        self.max_velocity = torch.tensor([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], device=self._device) # true for real ur5

        self._num_observation = 0 #107
        for item in self.config['training']['observations']['items']:
            self._num_observation += item['dimensions'] * (item['history'] + 1)
        self._num_action = 6 # 6 joint on ur5

        self.observation_space = None
        self.action_space = None

        
        BaseTask.__init__(self, name=name, offset=offset)

    def init_task(self):
        """didn't use this function"""

        # a few class buffers to store RL-related states
        self.ob = torch.zeros((self._num_envs, self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self._num_envs, self.num_agents, self.num_agents, self._num_observation), device=self._device)
        self.resets = torch.zeros((1), device=self._device) # all envs reset at the same time

        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_action), device=self._device)

        # rather than use for loop, batch processing can accelerate the process
        # franka_dof_targets should be initialized to be the same as the start config
        self.franka_dof_targets = torch.stack([torch.tensor(task.start_config, device = self._device) for task in self.current_tasks])

        # dof_limits = self._franka_list[0].get_dof_limits()
        # self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.collision = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        # add is_terminals to check if a individual robot terminate (collide or reach its target)
        # self.is_terminals = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        # define is_terminals to check if the env is terminal because of collision or success
        self.is_terminals = torch.zeros(self._num_envs, device=self._device)

        # for i in range(self._num_envs):
        #     for j in range(self.num_agents):
        #         self._franka_list[j].set_joint_positions(self.current_tasks[i].start_config[j])
        #         self._franka_list[j].target.set_local_pose(self.current_tasks[i].target_eff_pose[j])

        # use batch operation
        for i in range(self.num_agents):
            self._franka_list[i].set_local_poses(torch.tensor(current_task.start_config[i], device=self._device) for current_task in self.current_tasks)
            self._franka_list[i].set_joint_positions(self.franka_dof_targets[:, i])
            self._franka_list[i].target.set_local_pose(torch.tensor(current_task.target_eff_pose[i], device=self._device) for current_task in self.current_tasks)

    def set_up_scene(self, scene, replicate_physics=True) -> None:

        self.get_franka()
        self.get_target()
        
        # cloner class to create multiple envs
        collision_filter_global_paths = list()
        # if self._sim_config.task_config["sim"].get("add_ground_plane", True):
        self._ground_plane_path = "/World/defaultGroundPlane"
        collision_filter_global_paths.append(self._ground_plane_path)
        scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        # position of all envs
        self._env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics) 
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path, "/World/collisions", prim_paths, collision_filter_global_paths)


        self._franka_list=[]
        self._target_list=[]

        for i in range(4):

            # set franka1 in all envs as a UR5View
            franka = UR5MultiarmEnv(prim_paths_expr=self.default_base_env_path + "/.*/franka{}".format(i), name="franka{}_view".format(i),
                                ) # create a View for all the robots in all envs
            franka.ee = GeometryPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka{}/ee_link/ee".format(i), name="franka{}_view_ee".format(i))
            target = GeometryPrimView(prim_paths_expr=self.default_base_env_path + "/.*/target{}".format(i), name="franka{}_view_target".format(i))

            scene.add(franka)
            scene.add(franka.ee)
            scene.add(target)

            for link in franka.link_for_contact:
                scene.add(link)

            self._franka_list.append(franka)
            self._target_list.append(target)

        # set default pose of all robots are same to the origin of related env
            translations = torch.zeros((self._num_envs, 3), device=self._device)
            orientations = torch.zeros((self._num_envs, 4), device=self._device)
            orientations[:, 0] = 1.0
            franka.set_local_poses(translations, orientations)

        # set default camera viewport position and target
        self.set_initial_camera_params()

        # self.init_task()
        self.reset()

    def get_franka(self):
        

        usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/assets/ur5/ur5.usd"

        for i in range(4):
            ur5 = UR5(prim_path = self.default_zero_env_path + '/franka{}'.format(i), usd_path=usd_path)
            ee = VisualCylinder(prim_path=self.default_zero_env_path + "/franka{}/ee_link/ee".format(i), radius=0.02, height=0.1, name='UR5EE')
            # target = VisualCylinder(prim_path=self.default_zero_env_path + "/Franka/franka{}/target".format(i), radius=0.02, height=0.1,
            #                             color=np.array([0.8, 0.8, 0.8]),
            #                             # translation=target_pos if target_pos is not None else [0,0,-5],
            #                             # orientation=target_ori if target_ori is not None else None,
            #                             name='UR5Target')

    def get_target(self):
        for i in range(4):
            target = VisualCylinder(prim_path=self.default_zero_env_path + "/target{}".format(i), radius=0.02, height=0.1,
                                    color=np.array([0.8, 0.8, 0.8]),
                                    name='UR5Target')


    def set_initial_camera_params(self, camera_position=[5, 5, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def update_tasks(self):
        if self.mode == 'supervision':
            self.mode = 'normal'
            self.current_tasks = []
            for i in range(self._num_envs):
                current_task = self.taskloader.get_next_task()
                while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
                    current_task = self.taskloader.get_next_task()
                self.current_tasks.append(current_task)
        # no need to change to 'supervision' when all success
        elif self.mode == 'normal':
            self.mode = 'supervision'


    def reset(self):
        """
        Reset the environments.
        self.num_agents, self.collision, self.success, self.is_terminals
        self.ob, self.obs, self.actions

        Set : franka.set_local_pose(current_task.base_poses)
              franka.set_joint_positions(current_task.start_config)
              franka.set_joint_position_targets(current_task.start_config)
              franka.set_joint_velocities(zeros)

              franka.target.set_local_pose(current_task.target_eff_poses)

        
        """
        #updata tasks list
        self.update_tasks()
        self.num_agents=len(self.current_tasks[0].start_config)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self._num_action))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_agents, self._num_observation))

        self.collision = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        self.success = torch.zeros((self._num_envs), device=self._device)  
        self.is_terminals = torch.zeros(self._num_envs, device=self._device)    

        self.done = torch.zeros((self._num_envs, self.num_agents), device=self._device)

        self.ob = torch.zeros((self._num_envs, self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self._num_envs, self.num_agents, self.num_agents, self._num_observation), device=self._device)
        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_action), device=self._device)
        self.resets = torch.zeros((1), device=self._device)

        start_config = torch.stack([torch.tensor(task.start_config, device=self._device) for task in self.current_tasks])
        # start_config = torch.stack(torch.tensor(task.start_config, device = self._device) for task in self.current_tasks) # shape is (num_envs, num_agents, num_franka_dofs)
        self.franka_dof_targets = start_config
        dof_vel = torch.zeros((self._num_envs, self.num_agents, self.num_franka_dofs), device=self._device)


        """
        Reset the franka (positions, position targets, velocities, and local pose) to the start configuration.
        """
        # shape of base_pos is (num_envs, num_agents, 3), shape of base_ori is (num_envs, num_agents, 4), same to target_eff


        base_pos_list_envs = []
        base_ori_list_envs = []
        target_pos_list_envs = []
        target_ori_list_envs = []
        for current_task in self.current_tasks:
            # Use a list comprehension to gather base positions for all agents in the current task
            base_pos_list = [torch.tensor(current_task.base_poses[i][0], device=self._device) for i in range(self.num_agents)]
            base_ori_list = [torch.tensor(current_task.base_poses[i][1], device=self._device) for i in range(self.num_agents)]
            target_pos_list = [torch.tensor(current_task.target_eff_poses[i][0], device=self._device) for i in range(self.num_agents)]
            target_ori_list = [torch.tensor(current_task.target_eff_poses[i][1], device=self._device) for i in range(self.num_agents)]
            # Stack the base positions for the current task along dim=1
            base_pos_tensor = torch.stack(base_pos_list, dim=0)
            base_pos_list_envs.append(base_pos_tensor)

            base_ori_tensor = torch.stack(base_ori_list, dim=0)
            base_ori_list_envs.append(base_ori_tensor)

            target_pos_tensor = torch.stack(target_pos_list, dim=0)
            target_pos_list_envs.append(target_pos_tensor)

            target_ori_tensor = torch.stack(target_ori_list, dim=0)
            target_ori_list_envs.append(target_ori_tensor)

        # Stack the tensors for all tasks along dim=0
        # shape is [num_envs, num_agents, 3or4]
        base_pos = torch.stack(base_pos_list_envs, dim=0)
        base_ori = torch.stack(base_ori_list_envs, dim=0)
        base_ori = base_ori[:, :, [3,0,1,2]]
        target_eff_pos = torch.stack(target_pos_list_envs, dim=0)
        target_eff_ori = torch.stack(target_ori_list_envs, dim=0)
        target_eff_ori = target_eff_ori[:, :, [3,0,1,2]]

        for i in range(4):
            if i < self.num_agents:
                
                self._franka_list[i].set_joint_position_targets(start_config[:,i,:])
                self._franka_list[i].set_joint_positions(start_config[:,i,:])
                self._franka_list[i].set_joint_velocities(dof_vel[:,i,:])
                # self._franka_list[i].set_local_poses(translations = base_pos[:,i,:], orientations = base_ori[:,i,:])

                orientations = torch.zeros((self._num_envs,4),device=self._device)
                orientations[:,0] = 1.0
                # self._franka_list[i].world.set_local_poses(translations = torch.zeros((self._num_envs,3),device=self._device), orientations = orientations)
                self._franka_list[i].world.set_local_poses(translations = base_pos[:,i,:], orientations = base_ori[:,i,:])
                # self._franka_list[i].base_link.set_local_poses(translations = torch.zeros((self._num_envs,3),device=self._device), orientations = orientations)
                self._franka_list[i].base_link.set_local_poses(translations = base_pos[:,i,:], orientations = base_ori[:,i,:])
                self._target_list[i].set_local_poses(translations = target_eff_pos[:,i,:], orientations = target_eff_ori[:,i,:])
                # check local poses of robots and its config
                # print(str(self._franka_list[i].base_link.get_local_poses()) + 'and the configurations:')
                # for current_task in self.current_tasks:
                #     print(current_task.base_poses[i])
                # check if poses of target in simulation same as current_task.target_eff_poses
                # print(str(self._target_list[i].get_local_poses()) + 'and the configurations:')
                # for current_task in self.current_tasks:
                #     print(current_task.target_eff_poses[i]) 
                # after checking, target's local poses in simulation are same with the current_task.target_eff_poses
                # # check the base poses of robots in simulation with the current_task.base_poses
                # print(str(self._franka_list[i].get_local_poses()) + 'and the configurations:')
                # for current_task in self.current_tasks:
                #     print(current_task.base_poses[i])
            elif i >= self.num_agents:
                translations = torch.zeros(self._num_envs,3, device=self._device)
                translations[:,2] = -10
                # self._franka_list[i].set_local_poses(translations = translations)
                # self._franka_list[i].target.set_local_poses(translations = translations)
                self._franka_list[i].world.set_local_poses(translations = translations) 
                self._franka_list[i].base_link.set_local_poses(translations = translations) 
                self._target_list[i].set_local_poses(translations = translations)

        # test to solve the flashing cylinder problem
        # didn't work after testing
        # self._franka_list[i].ee.set_local_poses(translations = torch.zeros((self._num_envs,3),device=self._device), orientations = torch.zeros((self._num_envs,4),device=self._device))


        self.progress_buf = 0


    def pre_physics_step(self, actions) -> None: # actions should have size of (self._num_envs, self.num_agent, 6)

        actions = torch.tensor(actions).to(self._device)
        # set the actions in terminal envs to be 0s, and didn't add the data from terminal envs into replay_buffer
        # convert self.is_terminals into a torch.bool tensor 
        actions = torch.where((self.is_terminals==1).unsqueeze(-1).unsqueeze(-1), torch.zeros_like(actions), actions)

        # scaled_action should be action in (-1,+1) times max_velocity divided by simulation frequency
        # the following fomular should be thought over, the relationship with self.dt
        scaled_action = actions * self.max_velocity * self.dt * 3 # last scaler is a custom scaler to accelerate the training
        # targets = self.franka_dof_targets + self.dt * self.actions * self.action_scale # shape of self.num_agents*self.num_action, adapt self.franka_dof_targets based on its last value, making it changing smoothly
        targets = self.franka_dof_targets + scaled_action 
        self.franka_dof_targets[:] = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)
        # not certain about the indices
        # for i in range(self._num_envs):
        for i in range(self.num_agents):
            self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[:, i, :]) 
            # check the base poses of robots in simulation with the current_task.base_poses
            # print(str(self._franka_list[i].get_local_poses()) + 'and the configurations:')
            # for current_task in self.current_tasks:
            #     print(current_task.base_poses[i])
            # after reset, different

        # for i in range(4):
        #     print('poses of robot{} ee is :'.format(i) + str(self._franka_list[i].ee.get_world_poses()))
        #     print('poses of robot{} target is :'.format(i) + str(self._target_list[i].get_world_poses()))
        self.progress_buf += 1

        # test
        # targets_pos_list = []
        # for i, agent in enumerate(self._franka_list):
        #     target_pos = agent.target.get_world_poses()[0]
        #     target_pos_list.append(targets_pos)
        # targets_pos = torch.stack(target_pos_list, dim=0)




    def get_observation(self, this_franka):
        """
        shape of self.ob is (self._num_envs, self.num_agents, self._num_observation)
        """
        pos = np.array(this_franka.get_world_poses()[0]) # the base position of this_franka
        sorted_franka_list = sorted(self._franka_list[:self.num_agents], reverse=True, key=lambda agent: 
                                    np.linalg.norm(pos - np.array(agent.get_world_poses()[0]))) # get_world_poses() should return tensor with two element: position and orientation
        # ob = torch.zeros((self.num_agents, self._num_observation), device = self._device) 
        for i, agent in enumerate(sorted_franka_list[0:self.num_agents]):

            dof_pos = agent.get_joint_positions()
            # set the pos of ee identical to the pos of ee_link
            ee_pos, ee_rot = agent.ee_link.get_world_poses()[0], agent.ee_link.get_world_poses()[1]

            # positions = ee_pos.squeeze()
            # orientations = ee_rot.squeeze()
            agent.ee.set_world_poses(positions = ee_pos, orientations = ee_rot) # or set local poses to be all 0 so that same with parent ee_link

            # local_pose is the target pos relative to the base of robot
            # target_eff_pose = agent.target.get_world_poses()
            target_eff_pose = self._target_list[i].get_local_poses()
            # target_eff_pose = torch.tensor(np.concatenate(target_eff_pose, dim=0))
            target_eff_pose = torch.cat(target_eff_pose, dim=1)
            target_eff_pose = torch.cat([target_eff_pose, target_eff_pose], dim=1) # observation contains historical frame of target_eff_pose
            # goal_config = agent.goal_config

            # # link position is the center of mass, for simplification using pos of links here
            # link_position = agent.get_link_positions() # get link positions of links, 30 for ur5, why 30? reason: 10links*3xyz

            # here use center of mass
            link_position = agent.get_link_coms() 

            base_pose = agent.base_link.get_local_poses() # get the position of the base
            base_pose = torch.cat(base_pose, dim=-1).squeeze()

            if self.progress_buf == 1:
            # if self.ob == torch.zeros((self.num_agents, self._num_observation)): # if first step (no history yet)
             self.ob[:, i, 0:6] = dof_pos
             self.ob[:, i, 6:12] = dof_pos
             self.ob[:, i, 12:15] = ee_pos
             self.ob[:, i, 15:19] = ee_rot
             self.ob[:, i, 19:22] = ee_pos
             self.ob[:, i, 22:26] = ee_rot
             self.ob[:, i, 26:40] = target_eff_pose # 7*2
             self.ob[:, i, 40:70] = link_position
             self.ob[:, i, 70:100] = link_position
             self.ob[:, i, 100:107] = base_pose

            else:
             self.ob[:, i, 0:6] = self.ob[:, i, 6:12]
             self.ob[:, i, 6:12] = dof_pos
             self.ob[:, i, 12:15] = self.ob[:, i, 19:22]
             self.ob[:, i, 15:19] = self.ob[:, i, 22:26]
             self.ob[:, i, 19:22] = ee_pos
             self.ob[:, i, 22:26] = ee_rot
             self.ob[:, i, 26:40] = target_eff_pose # 7*2
             self.ob[:, i, 40:70] = self.ob[:, i, 70:100]
             self.ob[:, i, 70:100] = link_position
             self.ob[:, i, 100:107] = base_pose
        # print('end of one step \n')

        return self.ob # observation of a single franka (this_franka), shape of num_agents*ob of a single agent


    def get_observations(self):
        """
        shape of self.obs is (self._num_envs, self.num_agents, self.num_agents, self._num_observation)
        """
        
        # firstly sort the self._franka_list by base distance, furthest to closest, for each env

        # obs = torch.zeros((self._num_envs, self.num_agents, self.num_agents, self._num_observation), device = self._device)
        """
        computation should be slow with the following method, figure it out if possible to manipulate with torch operation
        """
        # for i in range(self._num_envs):
        for j, agent in enumerate(self._franka_list[0:self.num_agents]):
            self.obs[:, j, :, :] = self.get_observation(this_franka=agent)


        return self.obs # observation of the whole system, shape of num_agents*num_agents*ob of a single agent(107)

    def check_collision(self):


        for i in range(self.num_agents):
            for j, link in enumerate(self._franka_list[i].link_for_contact):
                if link.get_net_contact_forces() is not None:
                    contact_force = torch.norm(link.get_net_contact_forces(), dim=1).to(self._device)
                    self.collision[:, i] = torch.where(contact_force > 0.3, 1, self.collision[:, i])
                    """
                    set self.is_terminals to be 1 if collision happens in an env
                    """
                    self.is_terminals = torch.any(self.collision, dim=1)
                    # self.is_terminals = self.collision

                    # if contact:
                    #     print('collision happens in env:' + str(i) + ' with link at path:' + str(link.prims)) # or link.prims
                    #     self.collision[i,j] = 1  
                    #     # set the is_terminal to be 1
                    #     self.is_terminals[i, j] = 1
                
        return self.collision



    def all_reach_targets(self):

        indiv_reach_targets = self.indiv_reach_targets()
        all_reach_targets = torch.all(indiv_reach_targets, dim=1).int()

        """
        set self.is_terminals to be 1 if all agents in an env reach their targets
        """
        # self.is_terminals = torch.where(all_reach_targets == 1, 1, self.is_terminals)
        return all_reach_targets
    
    def indiv_reach_targets(self):
        indiv_reach_targets = torch.zeros((self._num_envs, self.num_agents), device = self._device)
        for i in range(self.num_agents):
            pos_delta = np.linalg.norm(self._franka_list[i].ee_link.get_world_poses()[0] - self._target_list[i].get_world_poses()[0], axis=1, keepdims=True)
            ori_delta = np.linalg.norm(self._franka_list[i].ee_link.get_world_poses()[1] - self._target_list[i].get_world_poses()[1], axis=1, keepdims=True)
        # if pos_delta < self.position_tolerance and ori_delta < self.orientation_tolerance:
        #     # the agent terminates if reaches its target
        #     self.is_terminals[self._franka_list.index(agent)] = 1
        #     return 1
        # else:
        #     return 0
            pos_delta = torch.from_numpy(pos_delta).to(self._device).squeeze(dim=-1)
            ori_delta = torch.from_numpy(ori_delta).to(self._device).squeeze(dim=-1)
            indiv_reach_targets[:,i] = torch.where((pos_delta < self.position_tolerance) & (ori_delta < self.orientation_tolerance), 1, 0)

        return indiv_reach_targets


    def calculate_metrics(self) -> None: # calculate the rewards in each env.step()

        reward = torch.zeros((self._num_envs, self.num_agents), device = self._device)

        
        collision_penalties = torch.zeros((self._num_envs, self.num_agents), device = self._device)
        if self.progress_buf > 1:
            self.check_collision()
            collision_penalties = torch.where(self.collision == 1, self.collision_penalty, 0)
            # set the is_terminal of env with collision to 1
            self.is_terminals = torch.where(self.collision.any(dim=1) == 1, 1, self.is_terminals)
            # for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            #     collision = self.check_collision(agent=agent)
            #     if collision == 1:
            #         collision_penalties[i] = self.collision_penalty # -0.05
            #     elif collision == 0:
            #         collision_penalties[i] = 0
            #     else:
            #         raise ValueError('The reading of the contact sensor makes no sense')

            # test below
            # if self.collision.any()==1:
            #     print('collision happens')


        indiv_reach_target_rewards = torch.zeros((self._num_envs, self.num_agents))
        indiv_reach_target = self.indiv_reach_targets()
        indiv_reach_target_rewards = torch.where(indiv_reach_target==1, self.indiv_reach_target_reward, 0)
        # for i, agent in enumerate(self._franka_list[0:self.num_agents]):
        #     # if i < self.num_agents:
        #     if self.indiv_reach_targets(agent=agent):
        #         indiv_reach_target_rewards[i] = self.indiv_reach_target_reward # 0.01
        #     elif self.indiv_reach_targets(agent=agent) == 0:
        #         indiv_reach_target_rewards[i] = 0
        #     else:
        #         raise ValueError('The agent should either reach its target or not')

        pos_rewards = np.zeros((self._num_envs, self.num_agents))
        ori_rewards = np.zeros((self._num_envs, self.num_agents))
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            # axis = 1 to ensure that shape of delta is num_envs
            pos_delta = np.linalg.norm(agent.ee_link.get_world_poses()[0] - self._target_list[i].get_world_poses()[0], axis = 1)
            ori_delta = np.linalg.norm(agent.ee_link.get_world_poses()[1] - self._target_list[i].get_world_poses()[1], axis = 1)

            # Smooth, continuous reward for getting closer to the target position
            pos_rewards[:,i] = np.exp(-pos_delta / self.position_tolerance)

            # Smooth, continuous reward for aligning orientation to the target
            ori_rewards[:,i] = np.exp(-ori_delta / self.orientation_tolerance)

        pos_rewards = torch.from_numpy(pos_rewards).to(self._device)
        ori_rewards = torch.from_numpy(ori_rewards).to(self._device)


        
        # if self.all_reach_targets():
        #     collectively_reach_targets_reward = np.full((self.num_agents, ), self.coorp_reach_target_reward) # 1
        # else:
        #     collectively_reach_targets_reward = np.zeros(self.num_agents)
        collectively_reach_targets_reward = torch.where(self.all_reach_targets() == 1, self.coorp_reach_target_reward, 0)
        self.success = torch.where(collectively_reach_targets_reward == self.coorp_reach_target_reward, 1, self.success)
        self.is_terminals = torch.where(collectively_reach_targets_reward == self.coorp_reach_target_reward, 1, self.is_terminals)

        # update self.done
        self.done = torch.where(self.collision == 1, 1, self.done)
        self.done = torch.where(indiv_reach_target == 1, 1, self.done)
        self.done = torch.ones_like(self.done) if self.progress_buf >= self._max_episode_length else self.done

        # test when any in self.is_terminals equal to 1
        # if self.is_terminals.any() == 1:
        #     print('some env terminates')

        franka_rewards_sum = \
            collision_penalties + indiv_reach_target_rewards +\
            collectively_reach_targets_reward.unsqueeze(dim=-1) \
            + pos_rewards + ori_rewards

        reward = franka_rewards_sum

        return reward

    def is_done(self):

        resets = 0
        # all envs either success or collide
        if torch.all(self.is_terminals == 1):
            resets = 1
            # print('end episode because of all envs success or collision')
            if torch.all(self.success == 1):
                print('end episode because of all envs success')
            elif torch.all(self.collision == 1):
                print('end episode because of all envs collision')
            

        # reset when reach max steps
        # resets = 1 if self.progress_buf >= self._max_episode_length else resets
        if self.progress_buf >= self._max_episode_length:
            resets = 1
            print('end episode because of max steps')

        self.resets = resets
        if self.resets == 1:
            print('progress_buf: ', self.progress_buf)

        return resets