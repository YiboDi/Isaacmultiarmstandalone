import sys
import time
# sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/exts')

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
from omni.isaac.core.prims import XFormPrimView, GeometryPrimView, RigidPrimView, RigidContactView

from gym import spaces
import numpy as np
import torch
# import math
from math import pi

# import sys 
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone')
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/robots')
from taskloader import TaskLoader

from thesis_utils import load_config

# from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim

# from ur5withEEandTarget import UR5withEEandTarget
from ur5 import UR5
from ur5_view import UR5View
from ur5_view_multienv import UR5MultiarmEnv
from omni.isaac.core.objects import VisualCylinder
import omni

from omni.isaac.core.utils.types import ArticulationActions


class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None, env=None) -> None:
        self.mode = 'supervision'
        self._env = env

        self.config = load_config(path='/home/dyb/Thesis/Isaacmultiarmstandalone/config/default.json')

        self.taskloader = TaskLoader(root_dir='/home/dyb/Thesis/tasks', shuffle=True)
        self._num_envs = 256
        self._env_spacing = 5

        self.dt = 1/60 # difference in time between two consecutive states or updates

        self.progress_buf = 0

        self.default_zero_env_path = '/World/envs/env_0'
        self.default_base_env_path = '/World/envs'

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self._device = "cuda"

        self.collision_penalty = -15
        # self.delta_pos_reward = 0
        # self.delta_ori_reward = 0
        self.activation_radius = 100
        self.indiv_reach_target_reward = 5
        self.coorp_reach_target_reward = 10
        self.position_tolerance = 0.40 # modify based on the experiment result
        self.orientation_tolerance = 0.25
        # self.position_tolerance = 0.05 # modify based on the experiment result
        # self.orientation_tolerance = 0.05

        self.num_franka_dofs = 6

        self._max_episode_length = 150

        self.dof_lower_limits = torch.tensor([-2 * pi, -2 * pi, -pi, -2 * pi, -2 * pi, -2 * pi], device=self._device)
        self.dof_upper_limits = torch.tensor([2 * pi, 2 * pi, pi, 2 * pi, 2 * pi, 2 * pi], device=self._device)

        self.success = torch.zeros((self._num_envs), device=self._device)

        self.max_velocity = torch.tensor([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], device=self._device) # true for real ur5

        self._num_observation = 47 #107, modified to 47 without link position
        # for item in self.config['training']['observations']['items']:
        #     self._num_observation += item['dimensions'] * (item['history'] + 1)
        self._num_action = 6 # 6 joint on ur5

        self.observation_space = None
        self.action_space = None

        self.max_ee_pos = torch.tensor([1.6, 1.6, 0.9], device=self._device)
        self.min_ee_pos = torch.tensor([-1.6, -1.6, 0.0], device=self._device)
        self.max_base_pos = torch.tensor([0.8, 0.8], device=self._device)
        self.min_base_pos = torch.tensor([-0.8, -0.8], device=self._device)
        # quaternion = torch.tensor([0, 0, 0, 0], device=self._device)
        # self.max_ee_pose = torch.cat([self.max_ee_pos, quaternion], dim=-1)
        # self.min_ee_pose = torch.cat([self.min_ee_pos, quaternion], dim=-1)
        # self.max_base_pose = torch.cat([self.max_base_pos, quaternion], dim=-1)
        # self.min_base_pose = torch.cat([self.min_base_pos, quaternion], dim=-1)
        #! consider range of joint config 
        self.max_joint = torch.tensor([3.6, 0.1, 2.9, 3.1, 4.0, 4.2], device=self._device)
        self.min_joint = torch.tensor([-6.2, -3.3, -1.8, -4.3, -4.7, -4.4], device=self._device)

        self.single_agent = False

        
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

        self.collision = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        # add is_terminals to check if a individual robot terminate (collide or reach its target)
        # define is_terminals to check if the env is terminal because of collision or success
        self.is_terminals = torch.zeros(self._num_envs, device=self._device)

        # use batch operation
        for i in range(self.num_agents):
            self._franka_list[i].set_world_poses(torch.tensor(current_task.start_config[i], device=self._device) for current_task in self.current_tasks)
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
        scene.add_default_ground_plane(prim_path=self._ground_plane_path) # comsume lots of time
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
            # franka.ee = GeometryPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka{}/ee_link/ee".format(i), name="franka{}_view_ee".format(i))
            target = GeometryPrimView(prim_paths_expr=self.default_base_env_path + "/.*/target{}".format(i), name="franka{}_view_target".format(i))
            franka.base_link = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka{}/base_link".format(i), name="franka{}_view_base_link".format(i))
            franka.links = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka{}/.*_link|tool0|world".format(i), name="franka{}_view_links".format(i))
            # franka.link_for_contact = RigidContactView(prim_paths_expr=self.default_base_env_path + "/.*/franka{}/.*_link".format(i), name="franka{}_view_link_for_contact".format(i), 
            # # track_contact_forces= True, prepare_contact_sensors=True,
            # filter_paths_expr = [self.default_base_env_path + "/.*/franka.*/.*_link"]
            # )

            scene.add(franka)
            # scene.add(franka.ee)
            scene.add(target)
            scene.add(franka.base_link)
            scene.add(franka.links)

            # franka.link_for_contact.initialize()
            # franka.link_for_contact.enable_rigid_body_physics()
            # mass = franka.link_for_contact.get_inv_masses()

            # for link in franka.link_for_contact:
            #     scene.add(link)

            self._franka_list.append(franka)
            self._target_list.append(target)

        # set default pose of all robots are same to the origin of related env
            translations = torch.zeros((self._num_envs, 3), device=self._device)
            orientations = torch.zeros((self._num_envs, 4), device=self._device)
            orientations[:, 0] = 1.0
            franka.set_world_poses(translations, orientations)

        # create a FrankaView for all robots under num_envs and num_agents
        self.frankaview = ArticulationView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*", name="frankas_view")
        self.frankaview.base_link = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/base_link", name="frankas_view_base_link")
        self.frankaview.links = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/.*_link|tool0|world", name="frankas_view_links")
        self.frankaview.link_for_contact = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/.*_link", name="frankas_view_link_for_contact", track_contact_forces= True)
        self.frankaview.ee_link = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/ee_link", name="frankas_view_ee_link")
        self.targetview = GeometryPrimView(prim_paths_expr=self.default_base_env_path + "/.*/target.*", name="targets_view")

        scene.add(self.frankaview)
        scene.add(self.targetview)
        scene.add(self.frankaview.base_link)
        scene.add(self.frankaview.links)
        scene.add(self.frankaview.link_for_contact)
        scene.add(self.frankaview.ee_link)
        
        # set default camera viewport position and target
        self.set_initial_camera_params()

        # self.init_task()
        self.reset()

    def get_franka(self):
        

        usd_path = "/home/dyb/Thesis/Isaacmultiarmstandalone/assets/ur5/ur5.usd"

        for i in range(4):
            ur5 = UR5(prim_path = self.default_zero_env_path + '/franka{}'.format(i), usd_path=usd_path)
            # ee = VisualCylinder(prim_path=self.default_zero_env_path + "/franka{}/ee_link/ee".format(i), radius=0.02, height=0.1, name='UR5EE')

    def get_target(self):
        for i in range(4):
            target = VisualCylinder(prim_path=self.default_zero_env_path + "/target{}".format(i), radius=0.02, height=0.1,
                                    color=np.array([0.8, 0.8, 0.8]),
                                    name='UR5Target')


    def set_initial_camera_params(self, camera_position=[5, 5, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def update_tasks(self, single_agent):
        if self.mode == 'supervision':
            self.mode = 'normal'
            self.current_tasks = []
            for i in range(self._num_envs):
                # current_task = self.taskloader.get_next_task()
                # while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
                #     current_task = self.taskloader.get_next_task()
                # self.current_tasks.append(current_task)
                current_task = self.taskloader.get_next_task()
                if single_agent:
                    while len(current_task.start_config) != 1: # test only environments with single robot
                        current_task = self.taskloader.get_next_task()
                else:
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
        self.update_tasks(self.single_agent)
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


        # method below involve more overhead due to the repeated tensor initializations
        start_config = [task.start_config for task in self.current_tasks]
        start_config = torch.tensor(start_config, device=self._device)

        self.franka_dof_targets = start_config
        dof_vel = torch.zeros((self._num_envs, self.num_agents, self.num_franka_dofs), device=self._device)


        """
        Reset the franka (positions, position targets, velocities, and local pose) to the start configuration.
        """
        # shape of base_pos is (num_envs, num_agents, 3), shape of base_ori is (num_envs, num_agents, 4), same to target_eff

        base_pos_list_envs, base_ori_list_envs = [], []
        target_pos_list_envs, target_ori_list_envs = [], []

        for current_task in self.current_tasks:
            # Append the positions and orientations directly without converting to tensors yet
            base_pos_list_envs += [current_task.base_poses[i][0] for i in range(self.num_agents)]
            base_ori_list_envs += [current_task.base_poses[i][1] for i in range(self.num_agents)]
            target_pos_list_envs += [current_task.target_eff_poses[i][0] for i in range(self.num_agents)]
            target_ori_list_envs += [current_task.target_eff_poses[i][1] for i in range(self.num_agents)]

        # Convert the lists of lists into tensors
        base_pos = torch.tensor(base_pos_list_envs, device=self._device).view(-1, self.num_agents, 3) #[num_envs, num_agents, 3]
        base_ori = torch.tensor(base_ori_list_envs, device=self._device).view(-1, self.num_agents, 4)[:, :, [3, 0, 1, 2]]
        target_eff_pos = torch.tensor(target_pos_list_envs, device=self._device).view(-1, self.num_agents, 3)
        target_eff_ori = torch.tensor(target_ori_list_envs, device=self._device).view(-1, self.num_agents, 4)[:, :, [3, 0, 1, 2]]

        for i in range(4):
            if i < self.num_agents:
                
                self._franka_list[i].set_joint_position_targets(start_config[:,i,:])
                self._franka_list[i].set_joint_positions(start_config[:,i,:])
                self._franka_list[i].set_joint_velocities(dof_vel[:,i,:])
                # self._franka_list[i].set_local_poses(translations = base_pos[:,i,:], orientations = base_ori[:,i,:])

                orientations = torch.zeros((self._num_envs,4),device=self._device)
                orientations[:,0] = 1.0

                tran = base_pos[:,i,:]
                orie = base_ori[:,i,:]
                
                self._franka_list[i].set_world_poses(positions = tran+self._env_pos, orientations = orie) # set base link local pose
                
                # self._franka_list[i].base_link.set_local_poses(translations = tran, orientations = orie)
                # test
                # pos = self._franka_list[i].get_local_poses()[0]
                # pos_link =  self._franka_list[i].base_link.get_local_poses()[0]
                # world_pos = self._franka_list[i].get_world_poses()[0]
                # world_pos_link =  self._franka_list[i].base_link.get_world_poses()[0]

                # print('base_pos from config '+ str(tran))
                # print('base_pos from sim '+ str(pos))
                # print('world_base_pos from sim '+ str(world_pos))
                # print('pos of base_link from sim '+ str(pos_link))
                # print('world pos of base_link from sim '+ str(world_pos_link))
                # print()


                trans, orien = target_eff_pos[:,i,:], target_eff_ori[:,i,:]
                self._target_list[i].set_world_poses(positions = trans+self._env_pos, orientations = orien)
                # test
                # pos = self._target_list[i].get_local_poses()[0]
                # world_pos = self._target_list[i].get_world_poses()[0]
                # print('target_pos from config '+ str(trans))
                # print('target_pos from sim '+ str(pos))
                # print('world_pos from sim '+ str(world_pos))
                # print()

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
                self._franka_list[i].set_world_poses(positions = translations+self._env_pos)
                # self._franka_list[i].target.set_local_poses(translations = translations)
                # self._franka_list[i].world.set_local_poses(translations = translations) 
                # self._franka_list[i].base_link.set_local_poses(translations = translations) 
                self._target_list[i].set_world_poses(positions = translations+self._env_pos)

        self.progress_buf = 0

        self.base_pos = base_pos

    def pre_physics_step(self, actions) -> None: # actions should have size of (self._num_envs, self.num_agent, 6)

        actions = actions.to(self._device)
        # set the actions in terminal envs to be 0s, and didn't add the data from terminal envs into replay_buffer
        # convert self.is_terminals into a torch.bool tensor 
        actions = torch.where((self.is_terminals==1).unsqueeze(-1).unsqueeze(-1), torch.zeros_like(actions), actions)

        # scaled_action should be action in (-1,+1) times max_velocity divided by simulation frequency
        # the following fomular should be thought over, the relationship with self.dt
        # scaled_action = actions * self.max_velocity * self.dt * 3 # last scaler is a custom scaler to accelerate the training
        # targets = self.franka_dof_targets + self.dt * self.actions * self.action_scale # shape of self.num_agents*self.num_action, adapt self.franka_dof_targets based on its last value, making it changing smoothly
        # targets = self.franka_dof_targets + scaled_action 
        # try directly set the scaled action to robot instead of accumulation
        # targets = scaled_action

        # scale the actions from (-1,1) back to joint range
        actions = (actions+1)/2 * (self.max_joint - self.min_joint) + self.min_joint

        targets = actions
        self.franka_dof_targets[:] = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)
        # not certain about the indices
        # for i in range(self._num_envs):
        for i in range(self.num_agents):
            # self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[:, i, :]) 
            # try apply_action()
            action = ArticulationActions(joint_positions=self.franka_dof_targets[:, i, :])
            self._franka_list[i].apply_action(action) 
            # also set the joint position directly to the action, simulating that we have a perfect controler
            # self._franka_list[i].set_joint_positions(self.franka_dof_targets[:, i, :]) 
            # check the base poses of robots in simulation with the current_task.base_poses
            # print(str(self._franka_list[i].get_local_poses()) + 'and the configurations:')
            # for current_task in self.current_tasks:
            #     print(current_task.base_poses[i])
            # after reset, different

        # for i in range(4):
        #     print('poses of robot{} ee is :'.format(i) + str(self._franka_list[i].ee.get_world_poses()))
        #     print('poses of robot{} target is :'.format(i) + str(self._target_list[i].get_world_poses()))
        self.actions = actions
        self.progress_buf += 1

        # test
        # targets_pos_list = []
        # for i, agent in enumerate(self._franka_list):
        #     target_pos = agent.target.get_world_poses()[0]
        #     target_pos_list.append(targets_pos)
        # targets_pos = torch.stack(target_pos_list, dim=0)


    def get_observation(self):
        # ob = [[] for _ in range(self.num_agents)]
        for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            dof_pos = agent.get_joint_positions(clone=False).to(self._device)

            # eepos_pre = agent.ee.get_world_poses()

            self.ee_pos, self.ee_rot = agent.ee_link.get_world_poses(clone=False) # move to cuda
            # # agent.ee.set_world_poses(positions = ee_pos, orientations = ee_rot)
            # # eelinkpos = agent.ee_link.get_world_poses()
            # # eepos_post = agent.ee.get_world_poses()

            # # print('ee pos pre: {}'.format(eepos_pre))
            # # print('ee pos post: {}'.format(eepos_post))
            # # print('ee link pos: {}'.format(eelinkpos))



            target_eff_pose = self._target_list[i].get_world_poses()
            target_eff_pose = torch.cat(target_eff_pose, dim=1).to(self._device)
            self.target_eff_pose = target_eff_pose  
            target_eff_pose = torch.cat([target_eff_pose, target_eff_pose], dim=1) # observation contains historical frame of target_eff_pose

            # # link_position = agent.get_link_coms() 
            # # link_position = agent.get_link_positions().to(self._device)
            link_position = agent.links.get_world_poses()[0].to(self._device)
            link_position = link_position.view(self._num_envs, 30)

            if self.progress_buf == 0 or 1:
                base_pose = agent.base_link.get_local_poses() # get the position of the base
                base_pose = agent.get_world_poses()
                base_pose = torch.cat(base_pose, dim=-1).squeeze().to(self._device)
            
            # if self.progress_buf == 1:
            # if self.ob == torch.zeros((self.num_agents, self._num_observation)): # if first step (no history yet)
                self.ob[:, i, 0:6] = dof_pos # joint_position
                self.ob[:, i, 6:12] = dof_pos 
                self.ob[:, i, 12:15] = self.ee_pos # world pos, need to transpose to pos to env
                self.ob[:, i, 15:19] = self.ee_rot
                self.ob[:, i, 19:22] = self.ee_pos
                self.ob[:, i, 22:26] = self.ee_rot
                self.ob[:, i, 26:40] = target_eff_pose # 7*2 # local pos to the env
                self.ob[:, i, 40:70] = link_position # should get world pos and transfer to env related pos
                self.ob[:, i, 70:100] = link_position
                self.ob[:, i, 100:107] = base_pose

            else:
                self.ob[:, i, 0:6] = self.ob[:, i, 6:12]
                self.ob[:, i, 6:12] = dof_pos
                self.ob[:, i, 12:15] = self.ob[:, i, 19:22]
                self.ob[:, i, 15:19] = self.ob[:, i, 22:26]
                self.ob[:, i, 19:22] = self.ee_pos
                self.ob[:, i, 22:26] = self.ee_rot
                # self.ob[:, i, 26:40] = target_eff_pose # 7*2
                self.ob[:, i, 40:70] = self.ob[:, i, 70:100]
                self.ob[:, i, 70:100] = link_position

        return self.ob

    def pre_observations(self): 
        dof_pos = self.frankaview.get_joint_positions(clone=False)
        dof_pos = dof_pos.view(self._num_envs, 4, 6)[:,:self.num_agents,:].to(self._device)
        self.dof_pos = dof_pos
        self.ee_pos, self.ee_rot = self.frankaview.ee_link.get_world_poses(clone=False)
        self.ee_pos = self.ee_pos.view(self._num_envs, 4, 3)[:,:self.num_agents,:].to(self._device)
        self.ee_rot = self.ee_rot.view(self._num_envs,4,4)[:,:self.num_agents,:].to(self._device)
        link_position = self.frankaview.links.get_world_poses()[0].to(self._device)
        link_position = link_position.view(self._num_envs, 4, 30)[:,:self.num_agents,:]

        if self.progress_buf == 0 or 1:

            base_pose = self.frankaview.get_world_poses()
            base_pose = torch.cat(base_pose, dim=-1).squeeze().to(self._device)
            base_pose = base_pose.view(self._num_envs, 4, 7)[:,:self.num_agents,:]

            target_eff_pose = self.targetview.get_world_poses()
            target_eff_pose = torch.cat(target_eff_pose, dim=-1).to(self._device)
            target_eff_pose = target_eff_pose.view(self._num_envs, 4, 7)[:,:self.num_agents,:]
            self.target_eff_pose = target_eff_pose
            target_eff_pose = torch.cat([target_eff_pose, target_eff_pose], dim=-1) # observation contains historical frame of target_eff_pose

            
            self.ob[:, :, 0:6] = dof_pos # joint_position
            self.ob[:, :, 6:12] = dof_pos 
            self.ob[:, :, 12:15] = self.ee_pos # world pos, need to transpose to pos to env
            self.ob[:, :, 15:19] = self.ee_rot
            self.ob[:, :, 19:22] = self.ee_pos
            self.ob[:, :, 22:26] = self.ee_rot
            self.ob[:, :, 26:40] = target_eff_pose # 7*2 # local pos to the env
            self.ob[:, :, 40:70] = link_position # should get world pos and transfer to env related pos
            self.ob[:, :, 70:100] = link_position
            self.ob[:, :, 100:107] = base_pose

        else:
            self.ob[:, :, 0:6] = self.ob[:, :, 6:12]
            self.ob[:, :, 6:12] = dof_pos
            self.ob[:, :, 12:15] = self.ob[:, :, 19:22]
            self.ob[:, :, 15:19] = self.ob[:, :, 22:26]
            self.ob[:, :, 19:22] = self.ee_pos
            self.ob[:, :, 22:26] = self.ee_rot
            # self.ob[:, :, 26:40] = target_eff_pose # 7*2
            self.ob[:, :, 40:70] = self.ob[:, :, 70:100]
            self.ob[:, :, 70:100] = link_position

        return self.ob
    
    def pre_observations_nolinks(self): 
        dof_pos = self.frankaview.get_joint_positions(clone=False)
        dof_pos = dof_pos.view(self._num_envs, 4, 6)[:,:self.num_agents,:].to(self._device)
        self.dof_pos = dof_pos # self.dof_pos is the real joint position of the robot 
        
        self.ee_pos, self.ee_rot = self.frankaview.ee_link.get_world_poses(clone=False)
        self.ee_pos = self.ee_pos.view(self._num_envs, 4, 3)[:,:self.num_agents,:].to(self._device)
        self.ee_rot = self.ee_rot.view(self._num_envs,4,4)[:,:self.num_agents,:].to(self._device)
        # link_position = self.frankaview.links.get_world_poses()[0].to(self._device)
        # link_position = link_position.view(self._num_envs, 4, 30)[:,:self.num_agents,:]

        # normalization
        self.ee_pos = 2*(self.ee_pos - self.min_ee_pos)/(self.max_ee_pos - self.min_ee_pos) - 1
        # self.ee_rot = (self.ee_rot - f)/(self.max_ee_rot - self.min_ee_rot)
        dof_pos = 2*(dof_pos - self.min_joint)/(self.max_joint - self.min_joint) - 1

        # self.dof_pos = dof_pos

        if self.progress_buf <= 1:

            base_pose = self.frankaview.get_world_poses()
            # # normalization
            # base_pose[0] = 2 * (base_pose[0] - self.min_base_pos)/(self.max_base_pos - self.min_base_pos) - 1
            base_pose = torch.cat(base_pose, dim=-1).squeeze().to(self._device)
            base_pose = base_pose.view(self._num_envs, 4, 7)[:,:self.num_agents,:]
            # normalization of x and y
            base_pose[:,:,:2] = 2*(base_pose[:,:,:2] - self.min_base_pos)/(self.max_base_pos - self.min_base_pos) - 1

            target_eff_pose = self.targetview.get_world_poses()
            # # normalization
            # target_eff_pose[0] = 2 * (target_eff_pose[0] - self.min_ee_pos)/(self.max_ee_pos - self.min_ee_pos) - 1
            target_eff_pose = torch.cat(target_eff_pose, dim=-1).to(self._device)
            target_eff_pose = target_eff_pose.view(self._num_envs, 4, 7)[:,:self.num_agents,:]
            # normalization
            target_eff_pose[:,:,:3] = 2*(target_eff_pose[:,:,:3] - self.min_ee_pos)/(self.max_ee_pos - self.min_ee_pos) - 1
            self.target_eff_pose = target_eff_pose
            target_eff_pose = torch.cat([target_eff_pose, target_eff_pose], dim=-1) # observation contains historical frame of target_eff_pose

            # normalization

            
            self.ob[:, :, 0:6] = dof_pos # joint_position
            self.ob[:, :, 6:12] = dof_pos 
            self.ob[:, :, 12:15] = self.ee_pos # world pos, need to transpose to pos to env
            self.ob[:, :, 15:19] = self.ee_rot
            self.ob[:, :, 19:22] = self.ee_pos
            self.ob[:, :, 22:26] = self.ee_rot
            self.ob[:, :, 26:40] = target_eff_pose # 7*2 # local pos to the env
            # self.ob[:, :, 40:70] = link_position # should get world pos and transfer to env related pos
            # self.ob[:, :, 70:100] = link_position
            self.ob[:, :, 40:47] = base_pose

        else:
            self.ob[:, :, 0:6] = self.ob[:, :, 6:12]
            self.ob[:, :, 6:12] = dof_pos
            self.ob[:, :, 12:15] = self.ob[:, :, 19:22]
            self.ob[:, :, 15:19] = self.ob[:, :, 22:26]
            self.ob[:, :, 19:22] = self.ee_pos
            self.ob[:, :, 22:26] = self.ee_rot
            # self.ob[:, :, 26:40] = target_eff_pose # 7*2
            # self.ob[:, :, 40:70] = self.ob[:, :, 70:100]
            # self.ob[:, :, 70:100] = link_position

        return self.ob
    
    def get_observations(self):
        """
        shape of self.obs is (self._num_envs, self.num_agents, self.num_agents, self._num_observation)
        """
        obs_start = time.time()
        # firstly sort the self._franka_list by base distance, furthest to closest, for each env
        ob_start = time.time()
        # self.get_observation() # get dof_pos 
        self.pre_observations_nolinks() # get other ob, give up after testing
        ob_end = time.time()
        print('get_observation time:', ob_end - ob_start)

        self.obs = self.ob.unsqueeze(1).expand(-1,self.num_agents, -1, -1)

        distance = torch.cdist(self.base_pos, self.base_pos) # self.base_pose has shape of [n_e,n_a,3], dis has shape of [n_e,n_a,n_a]
        sorted_index = distance.argsort(descending = True).to(self._device) #[n_e,n_a,n_a]
        expanded_index = sorted_index.unsqueeze(-1).expand(-1,-1, -1, self._num_observation).to(self._device) 

        self.obs = torch.gather(self.obs, dim=2, index=expanded_index)

        obs_end = time.time()
        print('get_observations time:', obs_end - obs_start)

        return self.obs




    # def check_collision(self):


    #     for i in range(self.num_agents):
    #         # for j, link in enumerate(self._franka_list[i].link_for_contact):
    #         if self._franka_list[i].link_for_contact.get_net_contact_forces() is not None:
    #             contact_force = torch.norm(self._franka_list[i].link_for_contact.get_net_contact_forces(), dim=1).to(self._device)
    #             self.collision[:, i] = torch.where(contact_force > 0.3, 1, self.collision[:, i])
    #             """
    #             set self.is_terminals to be 1 if collision happens in an env
    #             """
    #             self.is_terminals = torch.any(self.collision, dim=1)
    #             # self.is_terminals = self.collision

    #             # if contact:
    #             #     print('collision happens in env:' + str(i) + ' with link at path:' + str(link.prims)) # or link.prims
    #             #     self.collision[i,j] = 1  
    #             #     # set the is_terminal to be 1
    #             #     self.is_terminals[i, j] = 1
                
    #     return self.collision

#! TODO
    def check_collision(self):
        # if self.frankaview.link_for_contact.get_net_contact_forces() is not None:
        contact_force = self.frankaview.link_for_contact.get_net_contact_forces()
        contact_force = torch.norm(contact_force, dim=-1).to(self._device)
        contact_force = contact_force.view(self._num_envs, 4, 8)[:, :self.num_agents, :]
        self.collision = torch.where(torch.any(contact_force, dim = -1) > 0.3, 1, 0) # check if any link of robot is contacted
        # print('collision happens with link at path:' + str(self.frankaview.link_for_contact.prims))
        """
        set self.is_terminals to be 1 if collision happens in an env, in calculate_metrics()
        """
        # self.is_terminals = torch.where(self.collision == 1, 1, self.is_terminals)



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
        # for i in range(self.num_agents):
            # pos_delta = np.linalg.norm(self._franka_list[i].ee_link.get_world_poses()[0] - self._target_list[i].get_world_poses()[0], axis=1, keepdims=True)
            # ori_delta = np.linalg.norm(self._franka_list[i].ee_link.get_world_poses()[1] - self._target_list[i].get_world_poses()[1], axis=1, keepdims=True)
        # ee_pos = self.frankaview.ee_link.get_world_poses()[0].to(self._device).view(self._num_envs, 4, 3)[:,:self.num_agents,:]
        # target_pos = self.targetview.get_world_poses()[0].to(self._device).view(self._num_envs, 4, 3)[:,:self.num_agents,:]
        # self.pos_delta = torch.norm(self.ob[:, :, 19:22] - self.ob[:, :, 26:29], dim=-1, keepdim=True).squeeze(dim=-1)
        self.pos_delta = torch.norm(self.ee_pos - self.target_eff_pose[:,:,:3], dim=-1, keepdim=True).squeeze(dim=-1)

        # ee_ori = self.frankaview.ee_link.get_world_poses()[1].to(self._device).view(self._num_envs, 4, 4)[:,:self.num_agents,:]
        # target_ori = self.targetview.get_world_poses()[1].to(self._device).view(self._num_envs, 4, 4)[:,:self.num_agents,:]
        # self.ori_delta = torch.norm(self.ob[:, :, 22:26] - self.ob[:, :, 29:33], dim=-1, keepdim=True).squeeze(dim=-1)
        # self.ori_delta = torch.norm(self.ee_rot - self.target_eff_pose[:,:,3:], dim=-1, keepdim=True).squeeze(dim=-1)
        self.ori_delta = self.quaternion_angle_difference(self.ee_rot, self.target_eff_pose[:,:,3:])


        # if pos_delta < self.position_tolerance and ori_delta < self.orientation_tolerance:
        #     # the agent terminates if reaches its target
        #     self.is_terminals[self._franka_list.index(agent)] = 1
        #     return 1
        # else:
        #     return 0
            # pos_delta = torch.from_numpy(pos_delta).to(self._device).squeeze(dim=-1)
            # ori_delta = torch.from_numpy(ori_delta).to(self._device).squeeze(dim=-1)
        indiv_reach_targets[:,:] = torch.where((self.pos_delta < self.position_tolerance) & (self.ori_delta < self.orientation_tolerance), 1, 0)

        return indiv_reach_targets
    
    def quaternion_angle_difference(self, q1, q2):
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
        dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
        angle = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        return angle


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

        # pos_rewards = np.zeros((self._num_envs, self.num_agents))
        # ori_rewards = np.zeros((self._num_envs, self.num_agents))
        # for i, agent in enumerate(self._franka_list[0:self.num_agents]):
        #     # axis = 1 to ensure that shape of delta is num_envs
        #     pos_delta = np.linalg.norm(agent.ee_link.get_world_poses()[0] - self._target_list[i].get_world_poses()[0], axis = 1)
        #     ori_delta = np.linalg.norm(agent.ee_link.get_world_poses()[1] - self._target_list[i].get_world_poses()[1], axis = 1)

        #     # Smooth, continuous reward for getting closer to the target position
        #     pos_rewards[:,i] = np.exp(-pos_delta / self.position_tolerance)

        #     # Smooth, continuous reward for aligning orientation to the target
        #     ori_rewards[:,i] = np.exp(-ori_delta / self.orientation_tolerance)

        # pos_rewards = torch.from_numpy(pos_rewards).to(self._device)
        # ori_rewards = torch.from_numpy(ori_rewards).to(self._device)

        pos_rewards = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        ori_rewards = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        # for i, agent in enumerate(self._franka_list[0:self.num_agents]):
            # Get world poses for the end-effector link and the target, ensuring they are PyTorch tensors
        # agent_poses = self.frankaview.ee_link.get_world_poses()[0].to(self._device).view(self._num_envs, 4, 3)[:,:self.num_agents,:]
        # target_poses = self.targetview.get_world_poses()[0].to(self._device).view(self._num_envs, 4, 3)[:,:self.num_agents,:]

        # agent_oris = self.frankaview.ee_link.get_world_poses()[1].to(self._device).view(self._num_envs, 4, 4)[:,:self.num_agents,:]
        # target_oris = self.targetview.get_world_poses()[1].to(self._device).view(self._num_envs, 4, 4)[:,:self.num_agents,:]

        # Calculate positional and orientation deltas
        # pos_delta = torch.norm(self.ee_pos - self.target_eff_pose[:,:,:3], dim=-1)
        # ori_delta = torch.norm(self.ee_rot - self.target_eff_pose[:,:,4:], dim=-1)

        # Smooth, continuous reward for getting closer to the target position
        # pos_rewards[:, :] = torch.exp(-self.pos_delta / self.position_tolerance)
        pos_rewards[:, :] = -1.0 * self.pos_delta
        # Smooth, continuous reward for aligning orientation to the target
        # ori_rewards[:, :] = torch.exp(-self.ori_delta / self.orientation_tolerance)
        ori_rewards[:, :] = -0.5 * self.ori_delta


        
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
        # -1 + 1 +
        # 2
        # + (0, 1/e) + (0, 1/e)

        reward = franka_rewards_sum

        return reward # reward for each robot

    def is_done(self):

        resets = 0
        # all envs either success or collide
        if torch.all(self.is_terminals == 1):
            resets = 1
            # print('end episode because of all envs success or collision')
            if torch.all(self.success == 1):
                print('end episode because of all envs success')
            elif torch.all(torch.any(self.collision == 1, dim=-1)):
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