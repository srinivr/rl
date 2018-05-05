import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.scheduler.constant_scheduler import StepDecayScheduler
from collections import namedtuple
import numpy as np


class DQNAgent(BaseAgent):

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000, lr=1e-3, momentum=0.9,
                 criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99, epsilon_scheduler=StepDecayScheduler(),
                 epsilon_scheduler_use_steps=True, target_update_frequency=1e4, parameter_update_frequency=1,
                 grad_clamp=None, mb_size=32, replay_buffer_size=100000, replay_buffer_min_experience=None):

        self.mb_size = mb_size
        self.replay_buffer_size = replay_buffer_size
        if self.replay_buffer_size > 0:
            if replay_buffer_min_experience:
                assert replay_buffer_min_experience <= self.replay_buffer_size
                self.replay_buffer_min_experience = replay_buffer_min_experience
            else:
                self.replay_buffer_min_experience = self.mb_size
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.transitions = namedtuple('Transition', 'state action reward next_state done')
        super().__init__(model_class, model_params, rng, device, n_episodes, lr, momentum, criterion, optimizer, gamma,
                         epsilon_scheduler, epsilon_scheduler_use_steps, target_update_frequency,
                         parameter_update_frequency, grad_clamp)

    def learn(self, env):
        returns = []
        for ep in range(self.n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                action, o_, reward, done, info = self._action(env, o)
                ret += reward
                self.replay_buffer.insert(self.transitions(o, action, reward, o_, done))
                if self._is_gather_experience():
                    self.elapsed_steps += 1
                    continue
                states, actions, targets = self._get_batch()
                o = o_
                self._step_updates(states, actions, targets)
            self._episode_updates()
            returns.append(ret)
            if ep % 100 == 0 and len(returns) >= 100:
                print('mean prev 100 returns:', ep, ':', np.mean(returns[-100:]))

    def _get_batch(self):
        samples = self.replay_buffer.sample(self.mb_size)
        batch = self.transitions(*zip(*samples))
        non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), device=self.device)
        states = torch.tensor(batch.state, device=self.device, dtype=torch.float)
        actions = torch.tensor(batch.action, device=self.device, dtype=torch.long)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        non_final_states = torch.tensor([s for s, d in zip(batch.next_state, batch.done) if not d],
                                        device=self.device, dtype=torch.float)
        targets = torch.zeros(self.mb_size, device=self.device)
        self.model_target.eval()
        targets[non_final_mask] += self.gamma * self.model_target(non_final_states).max(1)[0].detach()
        targets += rewards
        return states, actions, targets

    def _is_gather_experience(self):
        return len(self.replay_buffer) < self.replay_buffer_min_experience

    def get_epsilon(self):
        if self._is_gather_experience():
            return 1. # np.random.random() < 1 always; hence do random action until minimum experience
        epsilon = self.epsilon_scheduler.get_epsilon(self.elapsed_steps) if self.epsilon_scheduler_use_steps \
            else self.epsilon_scheduler.get_epsilon(self.elapsed_episodes)
        return epsilon
