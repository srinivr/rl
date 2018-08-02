from gym.core import Wrapper
import numpy as np
from tensorboardX import SummaryWriter


class TrainLossWrapper(Wrapper):
    def __init__(self, env, experiment_id, writer):
        Wrapper.__init__(self, env)
        # returns calc
        self.rewards = []
        self.episodes = 0
        self.experiment_id = experiment_id
        self.steps = 0
        self.writer = TrainLossWrapper.writer # SummaryWriter(comment='PushWrapped_env')# +str(experiment_id))

    def reset(self):
        self.episodes += 1
        if self.experiment_id is None or self.writer is None:
            print('something is weird')
        if self.episodes == 301:
            self.episodes = 1
            print('mean train loss env_'+str(self.experiment_id)+' :', np.sum(self.rewards[-100])/100.0)
            t_sum = np.sum(self.rewards[-100])/100.0
            self.writer.add_scalar('data/train_reward',t_sum ,
                                   (self.steps * 16) + self.experiment_id)  # TODO change 16
            self.rewards = []

        return self.env.reset()

    def step(self, action):
        self.steps += 1
        s, r, d, i = self.env.step(action)
        self.rewards.append(r)
        return s, r, d, i
