from multiarm_task import MultiarmTask

class MultiarmSupervision(MultiarmTask):
    def __init__(self, name="Multiarm"):
        super().__init__(name=name)
        self.mode = 'normal'
        # self.mode = 'supervision' # this is set to be supervision because the env will be reset at initialzation, so first task will run in normal mode.
        self.success = False


    def update_task(self): # this function will be called when reset, so condition should represent the mode in the following episode
        # return super().update_task()
        if self.mode == 'normal' and self.success == False:
            super().update_task()
            self.mode = 'supervision'
        elif self.mode == 'normal' and self.success == True:
            self.success = False
            super().update_task()
        elif self.mode =='supervision':
            # super().update_task()
            self.mode = 'normal'

    # def reset(self):
    #     super().reset()
    #     if self.mode == 'normal' and self.success:
    #         self.success = False
    #     elif self.mode =='normal' and not self.success:
    #         self.mode ='supervision'
    #     elif self.mode =='supervision':
    #         self.mode = 'normal'
    #         self.success = False
            
    def is_done(self):

        resets = 0
        # resets = torch.where(self.check_collision(), 1, resets)
        for i,agent in enumerate(self._franka_list[0:self.num_agents]):
            # if i < self.num_agents:
            collision = self.check_collision(agent=agent)
            if collision == 1:
                resets = 1
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        resets = 1 if self.progress_buf >= self._max_episode_length else resets

        self.resets = resets

        return resets