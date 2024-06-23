


class supervision():
    def __init__(self, offset=None) -> None:
        # self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')
        # self.taskloader = TaskLoader(root_dir='/home/dyb/Thesis/tasks', shuffle=True)
        # self.current_task = self.taskloader.get_next_task()
        # self.dt = 1/60 # difference in time between two consecutive states or updates
        # self.progress_buf = 0
        self.max_action = 1.0
        self.joint_tolerance = 0.2

    def act_expert(self, env):
        expert_waypoints = env.load_expert_waypoints_for_task(
                    task_id=env._task.current_task.id)

        # if next_wp_idx == None:
        #     next_wp_idx = 0
        # curr_j = env._task.get_joint_positions() # change franka_list
            
        # load curr_j from Isaac sim    
        curr_j = torch.empty(env._task.num_agents * 6,) 
        for i, agent in enumerate(env._task._franka_list[0:env._task.num_agents]):
            dof = agent.get_joint_positions()
            curr_j[i*6:i*6+6] = dof.clone()
        curr_j = curr_j.numpy()

        # find the nearest wp in expert_waypoints and set it as next_wp and its idx to be next_wp_idx
        result = [(idx, norm(wp - curr_j))
                for idx, wp in enumerate(expert_waypoints)]
        result.sort(
            key=lambda v: v[1])
        # for i in range(5):
        #     print('waypoint{} is:'.format(i)+str(expert_waypoints[i]))
        next_wp_idx = result[0][0] + 1 if result[0][0] < len(expert_waypoints)-1 else result[0][0]# ensure that the initial next_wp is next to curr_j

        # if next_wp is too close to curr_j, check next one in expert_waypoint. when condition is satisfied, initialize the target_j as next_wp.
        # if dis between next_wp and curr_j smaller than threshold (here 0.01), then we regard robots are current in this pose, so check next wp.
        # dis = curr_j - expert_waypoints[next_wp_idx] # dis check if robots already at this pose
        # seem 0.01 is too small
        # while dis.any() < 0.01:
        while np.all(curr_j - expert_waypoints[next_wp_idx] < 0.01):
            # next_wp_idx += 1
            if next_wp_idx >= len(expert_waypoints) - 1:
                break
            next_wp_idx += 1
            
        target_wp_idx = next_wp_idx
        next_dir_j = expert_waypoints[next_wp_idx] - curr_j
        # find the most far away expert wp (target_j) in dataset that fullfil the conditions(tolerance config)
        # comfirm the final target_j in this step
        while True: 
            target_j = expert_waypoints[target_wp_idx]
            target_dir_j = target_j - curr_j
            # max_action = 0.5 # max action per simulation step, default value
            # max_action = 1.0 # set max_action to be 1 to aglin with the output of policy network
            # joint_tolerance = 0.1 # default
            # joint_tolerance = 0.2

            if target_wp_idx < len(expert_waypoints) - 1 and \
                all([delta_j < self.max_action for delta_j in abs(target_dir_j)])\
                    and angle(next_dir_j, target_dir_j) < self.joint_tolerance:
                target_wp_idx += 1
            else:
                break
        # if next_wp_idx (which is nearest waypoint to the curr_j) is the last one, change the mode to normal
        # if next_wp_idx < len(expert_waypoints) - 1:
        #     next_wp_idx += 1
        # # actions = target_j - curr_j
        # # env.step(actions)
            
        # # so when else, should reset the task, modify
        # else:
        #     mode = 'normal'

        #calculate the actions based on the waypoints
        actions = target_j - curr_j
        # actions should be target_j instead of target_j - curr_j?
        # actions = target_j
        actions = actions.reshape((env._task.num_agents, 6))
        actions = torch.from_numpy(actions).clone()
        return actions
