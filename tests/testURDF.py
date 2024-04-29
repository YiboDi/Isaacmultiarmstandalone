import pybullet as p

p.connect(p.GUI)
robot = p.loadURDF("/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/assets/ur5/ur5.urdf", useFixedBase=True)

num_joints = p.getNumJoints(robot)

for joint in range(num_joints):
    joint_info = p.getJointInfo(robot, joint)
    joint_name = joint_info[1].decode('utf-8')
    parent_index = joint_info[16]
    if parent_index == -1:
        parent_name = "world"
    else:
        parent_name = p.getJointInfo(robot, parent_index)[1].decode('utf-8')

    print(f"Joint {joint_name} connects to {parent_name}")