from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from ur5 import UR5
from omni.isaac.core import World
from omniisaacgymenvs.robots.articulations.franka import Franka
from omni.isaac.core.utils.nucleus import get_assets_root_path

if __name__ == "__main__":

    world = World()

    world.scene.add_default_ground_plane()
    usd_path = "Di_custom/multiarmRL/assets/ur5/ur5.usd"

    ur5 = UR5(prim_path="/World/ur5",  
            # translation=pos[i], 
        # orientation=orientation, 
            usd_path=usd_path, 
            # default_dof_pos=default_dof_pos,
            name = 'ur5')
    
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