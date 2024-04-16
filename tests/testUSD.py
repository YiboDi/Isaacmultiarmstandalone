import sys
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/robots')

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from ur5 import UR5
from ur5_view import UR5View
from omni.isaac.core import World
from omniisaacgymenvs.robots.articulations.franka import Franka
from omni.isaac.core.utils.nucleus import get_assets_root_path

import torch

if __name__ == "__main__":

    world = World()

    world.scene.add_default_ground_plane()
    usd_path = "Di_custom/multiarmRL/assets/ur5/ur5.usd"

    UR5(prim_path="/World/ur5",  
            # translation=pos[i], 
        # orientation=orientation, 
            usd_path=usd_path, 
            # default_dof_pos=default_dof_pos,
            name = 'ur5')
    
    ur5 = UR5View(prim_paths_expr="/World/ur5")
    
    pos = ur5.get_local_poses()
    print('pos before reset' + str(pos))

    ur5.set_local_poses(torch.tensor([1, 1, 0]).unsqueeze(dim=0), torch.tensor([0, 0, 0, 1]).unsqueeze(dim=0))
    print('pos after reset' + str(ur5.get_local_poses()))
    
    
    # ur5_test = UR5(prim_path="/World/ur5test",  
    #         # translation=pos[i], 
    #     # orientation=orientation, 
    #         usd_path=usd_path, 
    #         # default_dof_pos=default_dof_pos,
    #         name = 'ur5_test')
    # assets_root_path = get_assets_root_path()
    # usd_path = assets_root_path + '/Isaac/Robots/Franka/franka_instanceable.usd'

    # franka = Franka(prim_path='/World/franka',
    #                 usd_path=usd_path,
    #                 name= 'franka')
    
    
    while True:
        world.step()