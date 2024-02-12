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
from omni.isaac.core.objects import VisualCylinder



class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None) -> None:

        self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')

        self.taskloader = TaskLoader(root_dir='/home/tp2/papers/multiarm_dataset/tasks', shuffle=True)
        # self.current_task = self.taskloader.get_next_task()
        self.current_tasks = []
        for i in range(self._num_envs):
            current_task = self.taskloader.get_next_task()
            while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
                current_task = self.taskloader.get_next_task()

            self.current_tasks.append(self.taskloader.get_next_task())

        self.num_agents=len(self.current_tasks[0].start_config)

        self.dt = 1/60 # difference in time between two consecutive states or updates

        self.progress_buf = 0

        self.default_zero_env_path = '/World/envs/env_0'
        self.default_base_env_path = '/World/envs'

        self._num_envs = 3
        self._env_spacing = 10

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

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

        self._max_episode_length = 300
        # self._max_episode_length = 500 # from config

        self.dof_lower_limits = torch.tensor([-2 * pi, -2 * pi, -pi, -2 * pi, -2 * pi, -2 * pi], device=self._device)
        self.dof_upper_limits = torch.tensor([2 * pi, 2 * pi, pi, 2 * pi, 2 * pi, 2 * pi], device=self._device)

        self.success = 0

        self.max_velocity = torch.tensor([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], device=self._device) # true for real ur5

        BaseTask.__init__(self, name=name, offset=offset)

    def init_task(self):

        #values used for defining RL buffers for multiagent and dynamic
        self._num_observation = 0 
        for item in self.config['training']['observations']['items']:
            self._num_observation += item['dimensions'] * (item['history'] + 1)


        # values used for defining RL buffers

        self._num_action = 6 # 6 joint on ur5
        self._num_actions = self._num_action * self.current_task.ur5_count # num of actions in single env

        # a few class buffers to store RL-related states
        self.ob = torch.zeros((self._num_envs, self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self._num_envs, self.num_agents, self.num_agents, self._num_observation), device=self._device)
        self.resets = torch.zeros((1), device=self._device) # all envs reset at the same time

        # set the action and observation space for RL

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self._num_action))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_agents, self._num_observation))


        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_action), device=self._device)

        self.franka_dof_targets = torch.zeros((self._num_envs, self.num_agents, self.num_franka_dofs), device=self._device)
        for i in range(self._num_envs):
            self.franka_dof_targets[i] = torch.tensor(self.current_tasks[i].start_config, device=self._device)

        # dof_limits = self._franka_list[0].get_dof_limits()
        # self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)

        # add is_terminals to check if a individual robot terminate (collide or reach its target)
        self.is_terminals = torch.zeros((self._num_envs, self.num_agents), device=self._device)

        for i in range(self._num_envs):
            for j in range(self.num_agents):
                self._franka_list[j].set_joint_positions(self.current_tasks[i].start_config[j])
                self._franka_list[j].target.set_local_pose(self.current_tasks[i].target_eff_pose[j])

    def set_up_scene(self, scene, replicate_physics=True) -> None:

        self.get_franka()
        
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


        # self.get_franka()


        self._franka_list=[]

        for i in range(4):

            # if i < self.num_agents:
                
                # franka = UR5View(prim_paths_expr=self.default_base_env_path + "/.*/Franka/franka{}".format(i), name="franka{}_view".format(i),
                #                 ) # create a View for all the robots in all envs

            # # add the following variable in UR5View init
            #     # self._franka_list[-1].target_eff_pose = self.current_task.target_eff_poses[i] # shape of num_agents*(translation3+orientation4), precisely a list of num_agents * [3.4]
            #     # self._franka_list[-1].goal_config = self.current_task.goal_config[i] # goal joint states, shape same with above

            # elif i >= self.num_agents:
            #     franka = UR5View(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i),
            #                     )

            franka = UR5View(prim_paths_expr=self.default_base_env_path + "/.*/Franka/franka{}".format(i), name="franka{}_view".format(i),
                                ) # create a View for all the robots in all envs

            scene.add(franka)
            scene.add(franka.ee)
            scene.add(franka.target)

            for link in franka.link_for_contact:
                scene.add(link)

            self._franka_list.append(franka)

        # set default camera viewport position and target
        self.set_initial_camera_params()

        self.init_task()

    def get_franka(self):
        

        usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/assets/ur5/ur5.usd"
        # for a in range(self._num_envs):
        #     current_task = self.current_tasks[a]
        for i in range(4):
            if i < self.num_agents:
                position = torch.tensor(current_task.base_poses[i][0])
                orientation = torch.tensor(current_task.base_poses[i][1])
                orientation = orientation[[3,0,1,2]]
                default_dof_pos = current_task.start_config[i]
                # need to rethink about the target and ee
                # target_pos = current_task.target_eff_poses[i][0]
                # target_ori = current_task.target_eff_poses[i][1]
                # target_ori = target_ori[-1:] + target_ori[:-1]
                # ur5 = UR5withEEandTarget(prim_path= self.default_base_env_path + "env_{}/Franka/franka{}".format(a, i), translation=position, orientation=orientation, usd_path=usd_path, default_dof_pos=default_dof_pos,
                #                          target_ori=target_ori, target_pos=target_pos) 
                ur5 = UR5(prim_path = self.default_zero_env_path + '/Franka/franka{}'.format(i), usd_path=usd_path)

                
            elif i >= self.num_agents:
                ur5 = UR5(prim_path=self.default_zero_env_path + "/Franka/franka{}".format(i), translation=[0, 0, -10], 
                   usd_path=usd_path, 
                   )

    def set_initial_camera_params(self, camera_position=[5, 5, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def update_tasks(self):
        self.current_tasks = []
        for i in range(self._num_envs):
            current_task = self.taskloader.get_next_task()
            while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
                current_task = self.taskloader.get_next_task()
            self.current_tasks.append(current_task)


    def reset(self):
        #get the next task
        self.update_tasks()

        self.num_agents=len(self.current_tasks[0].start_config)

        self.success = torch.zeros((self._num_envs), device=self._device)
       

        self._num_actions = self._num_action * self.current_task.ur5_count

        self.ob = torch.zeros((self._num_envs, self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self._num_envs, self.num_agents, self.num_agents, self._num_observation), device=self._device)
        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_action), device=self._device)

        self.is_terminals = torch.zeros((self._num_envs, self.num_agents), device=self._device)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self._num_action))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_agents, self._num_observation))

        self.num_franka_dofs = 6 # same to self.num_action

        for i in range(self._num_envs):
            self.franka_dof_targets[i] = torch.tensor(self.current_tasks[i].start_config, device=self._device)

        dof_vel = torch.zeros((self._num_envs, self.num_agents, self.num_franka_dofs), device=self._device)

        base_poses = self.current_task.base_poses
        start_config = self.current_task.start_config
        # goal_config = self.current_task.goal_config
        target_eff_poses = self.current_task.target_eff_poses

        for i in range(self._num_envs):
            for j in range(4):
                if j < self.num_agents:
                    base_poses = self.current_tasks[i].base_poses
                    start_config = self.current_tasks[i].start_config
                    target_eff_poses = self.current_tasks[i].target_eff_poses

                    pos = torch.tensor(base_poses[j][0]).unsqueeze(0)
                    ori = torch.tensor(base_poses[j][1]).unsqueeze(0)
                    ori = ori[:,[3,0,1,2]]

                    
                    self._franka_list[j].set_local_pose(pos, ori, indices = torch.tensor(i))
                    self._franka_list[j].set_joint_position_targets(torch.tensor(start_config[j], device=self._device), indices=torch.tensor(i))
                    self._franka_list[j].set_joint_positions(torch.tensor(start_config[j]), indices=torch.tensor(i))
                    self._franka_list[j].set_joint_velocities(torch.tensor(dof_vel[j]), indices=torch.tensor(i))

                    position = target_eff_poses[i][j][0]
                    orientation = target_eff_poses[i][j][1]
                    orientation = orientation[-1:] + orientation[:-1]

                    self._franka_list[j].target.set_local_pose(position = position, orientation = orientation, indices = torch.tensor(i))
                elif j >= self.num_agents:
                    self._franka_list[j].set_local_poses(positions = torch.tensor([[0,0,-10]]), 
                                                         indices = torch.tensor(j)
                                                    #  indices = torch.tensor(0)
                                                     )

                    self._franka_list[j].target.set_world_pose(position = [0,0,-10])



        self.progress_buf = 0


    def pre_physics_step(self, actions) -> None:

        # if reset == 1:
        #     self.reset()

        actions = torch.tensor(actions).to(self._device)

        # scaled_action should be action in (-1,+1) times max_velocity divided by simulation frequency
        scaled_action = actions * self.max_velocity * self.dt * 3 # last scaler is a custom scaler to accelerate the training
        # targets = self.franka_dof_targets + self.dt * self.actions * self.action_scale # shape of self.num_agents*self.num_action, adapt self.franka_dof_targets based on its last value, making it changing smoothly
        targets = self.franka_dof_targets + scaled_action 
        self.franka_dof_targets[:] = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)
        for i in range(self._num_envs):
            for j in range(self.num_agents):
                self._franka_list[j].set_joint_position_targets(self.franka_dof_targets[i, j, :], indices = i) 

        self.progress_buf += 1

        # base_pos = self._franka_list[0].get_world_poses()
        # print(str(base_pos))

        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observation(self, this_franka):
        pos = np.array(this_franka.get_world_poses()[0]) # the base position of this_franka
        sorted_franka_list = sorted(self._franka_list[:self.num_agents], reverse=True, key=lambda agent: 
                                    np.linalg.norm(pos - np.array(agent.get_world_poses()[0]))) # get_world_poses() should return tensor with two element: position and orientation
        ob = torch.zeros((self.num_agents, self._num_observation), device = self._device) 
        for i, agent in enumerate(sorted_franka_list[0:self.num_agents]):

            dof_pos = agent.get_joint_positions()
            # set the pos of ee identical to the pos of ee_link
            
            ee_pos, ee_rot = agent.ee_link.get_world_poses()[0], agent.ee_link.get_world_poses()[1]

            # test
            # print('for robot{}:'.format(i))
            # # print('position of ee is :' + str(ee_pos) + '\n')
            # print('position of ee_link is :' + str(agent.ee_link.get_world_poses()) )
            # # print('difference value is :' + str(ee_pos - agent.ee_link.get_world_poses()[0]) + '\n')
            # print('position of target is :' + str(agent.target.get_world_pose()) )
            pos_diff = agent.ee_link.get_world_poses()[0] - agent.target.get_world_pose()[0]
            pos_delta = np.linalg.norm(pos_diff)
            ori_diff = agent.ee_link.get_world_poses()[1] - agent.target.get_world_pose()[1]
            ori_delta = np.linalg.norm(ori_diff)

            # print('position difference is :' + str(pos_delta) )
            # print('orientation difference is :' + str(ori_delta) )
            


            agent.ee.set_world_pose(position = ee_pos.squeeze(), orientation = ee_rot.squeeze())

            target_eff_pose = agent.target.get_world_pose()
            target_eff_pose = torch.tensor(np.concatenate(target_eff_pose))
            target_eff_pose = torch.cat([target_eff_pose, target_eff_pose]) # observation contains historical frame of 
            # goal_config = agent.goal_config

            # link position is the center of mass, for simplification using pos of links here
            link_position = agent.get_link_positions() # get link positions of links, 30 for ur5, why 30? reason: 10links*3xyz

            base_pose = agent.get_world_poses() # get the position of the base
            base_pose = torch.cat(base_pose, dim=-1).squeeze()

            if self.progress_buf == 1:
            # if self.ob == torch.zeros((self.num_agents, self._num_observation)): # if first step (no history yet)
                ob[i, 0:6] = dof_pos
                ob[i, 6:12] = dof_pos
                ob[i, 12:15] = ee_pos
                ob[i, 15:19] = ee_rot
                ob[i, 19:22] = ee_pos
                ob[i, 22:26] = ee_rot
                ob[i, 26:40] = target_eff_pose # 7*2
                ob[i, 40:70] = link_position
                ob[i, 70:100] = link_position
                ob[i, 100:107] = base_pose

            else:
                ob[i, 0:6] = self.ob[i, 6:12]
                ob[i, 6:12] = dof_pos
                ob[i, 12:15] = self.ob[i, 19:22]
                ob[i, 15:19] = self.ob[i, 22:26]
                ob[i, 19:22] = ee_pos
                ob[i, 22:26] = ee_rot
                ob[i, 26:40] = target_eff_pose # 7*2
                ob[i, 40:70] = self.ob[i, 70:100]
                ob[i, 70:100] = link_position
                ob[i, 100:107] = base_pose
        self.ob = ob.clone()
        # print('end of one step \n')

        return self.ob # observation of a single franka (this_franka), shape of num_agents*ob of a single agent


    def get_observations(self):
        # do not consider multi env temporally
        
        # firstly sort the self._franka_list by base distance, furthest to closest.

        obs = torch.zeros((self.num_agents, self.num_agents, self._num_observation), device = self._device)
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):

            
            obs[i, :, :] = self.get_observation(this_franka=agent)


            # hand_pos, hand_rot = agent._hands.get_world_poses()
            # dof_pos = agent.get_joint_positions()
            # # dof_vel = agent.get_joint_velocities()


            # if dof_pos.shape[0]+dof_vel.shape[0] != self._num_observation:
            #     raise ValueError('dim of observation does not match')
            
            # self.obs[i,:] = torch.cat(dof_pos, hand_pos, hand_rot) # shape of self.obs is num_robots * (num_joint_pos * num_joint_vel)
        self.obs = obs.clone()

        return self.obs # observation of the whole system, shape of num_agents*num_agents*ob of a single agent(107)

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