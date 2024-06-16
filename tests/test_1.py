from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})
import sys

sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/robots')

from ur5 import UR5
from ur5_view import UR5View
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.objects import VisualCylinder
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.articulations import ArticulationView

import numpy as np
from omni.isaac.core.utils.prims import get_prim_path
from omni.isaac.core.prims import RigidPrimView
import torch



if __name__ == "__main__":

    world = World()

    world.scene.add_default_ground_plane()
    # usd_path = "Di_custom/multiarmRL/assets/ur5/ur5.usd"
    usd_path = '/home/dyb/Thesis/Isaacmultiarmstandalone/assets/ur5/ur5.usd'

    # assets_root_path = get_assets_root_path()
    # usd_path = assets_root_path + "/Isaac/Robots/UR10/ur10.usd"

    ur5_list = []

    for i in range(1):
        pos1 = [0,0,0]
        pos2 = [0,1,0]
        pos = [pos1, pos2]
        # orientation = [0,0,0,0]
        default_dof_pos = [0,0,0,0,0,0]
        UR5(prim_path="/World/Franka/franka{}".format(i),  # do not work at first, solved after correct the path
                  translation=pos[i], 
                # orientation=orientation, 
                  usd_path=usd_path, default_dof_pos=default_dof_pos,
                  name = 'ur5_{}'.format(i)) 
        
        # ur5 = Franka(prim_path="/World/UR5",  # do not work
        #         #   translation=position, orientation=orientation, 
        #           usd_path=usd_path, 
        #         #   default_dof_pos=default_dof_pos
        #           ) 
        # cube = world.scene.add(DynamicCuboid(prim_path='/cube'))
        # create_prim(prim_path="/World/UR5", prim_type="Xform",  # work well
        #             # position=self._cartpole_position, 
        #             usd_path=usd_path)
        
        # ur5 = UR5View(prim_paths_expr="/World/UR5", name="ur5_view")
        ur5 = UR5View(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i),
                      target_pos=[1,1,1])
        ur5_list.append(ur5)
        # ur5_list[-1].ee_link = RigidPrimView(prim_paths_expr="/World/Franka/franka{}/ee_link".format(i))
        world.scene.add(ur5_list[-1])
        world.scene.add(ur5_list[-1].ee)
        # world.scene.add(ur5_list[-1].target)
        # ee = world.scene.add(VisualCylinder(prim_path ='/World/Franka/franka{}/ee_link/point'.format(i), radius=0.02, height=0.1, name = 'cylinder{}'.format(i)))

    # for i in range(1):
    #     assets_root_path = get_assets_root_path()
    #     usd_path = assets_root_path + '/Isaac/Robots/Franka/franka_instanceable.usd'
    #     # create_prim(prim_path='/World/franka', usd_path=usd_path)
    #     franka = Franka(prim_path='/World/franka', usd_path=usd_path)

    while True:
        # position, orientation = ur5.get_world_poses()
        # # position, orientation = cube.get_world_pose()
        # print('position is :' + str(position))
        # print('orientation is :' + str(orientation))
        # position,orientation = ee.get_world_pose()
        # pos = ur5.get_world_poses()
        
        # for i, agent in enumerate(ur5_list):
        for i in range(1000):
            world.step(render=True)
        
        pos,ori = ur5_list[0].get_world_poses()

        # ur5_list[0].set_world_poses(positions = pos + 1, orientations = ori)
        new_pos = ur5_list[0].get_world_poses()
            # print(str(agent.ee_link.get_coms()))
            # print(str(agent.ee_link.get_world_poses()))
            # link_pos = agent.get_link_positions()
            # contact = agent.get_contact_force_data()
            # print(str(contact))


        # print(str(pos))
        # print(str(pos[0] - np.array([1,1,1])))

        world.step(render=True)


