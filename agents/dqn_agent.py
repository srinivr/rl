import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.scheduler.decay_scheduler import StepDecayScheduler
from collections import namedtuple
import numpy as np


class DQNAgent(BaseAgent):

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000, n_eval_steps=100, lr=1e-3,
                 momentum=0.9, criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99,
                 epsilon_scheduler=StepDecayScheduler(), epsilon_scheduler_use_steps=True, target_update_steps=1e4,
                 parameter_update_frequency=1, grad_clamp=None, mb_size=32, replay_buffer_size=100000,
                 replay_buffer_min_experience=None):

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
        super().__init__(model_class, model_params, rng, device, n_episodes, n_eval_steps, lr, momentum, criterion,
                         optimizer, gamma, epsilon_scheduler, epsilon_scheduler_use_steps, target_update_steps,
                         parameter_update_frequency, grad_clamp)

    def learn(self, env):
        returns = []
        for ep in range(self.n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                action, o_, reward, done, info = self._get_action(env, o)
                ret += reward
                self.replay_buffer.insert(self.transitions(o, action, reward, o_, done))
                if self._is_gather_experience():
                    continue
                states, actions, targets = self._get_batch()
                self._step_updates(states, actions, targets)
                o = o_
            self._episode_updates()
            returns.append(ret)
            if (ep+1) % 100 == 0 and len(returns) >= 100:
                print('mean prev 100 returns:', ep, ':', np.mean(returns[-100:]))
                print('ep:', ep, end=' ')
                self._eval(env)
                returns = []

    def _get_batch(self):
        samples = self.replay_buffer.sample(self.mb_size)
        batch = self.transitions(*zip(*samples))  # https://stackoverflow.com/a/19343/3343043
        return super()._get_batch(batch.state, batch.action, batch.next_state, batch.reward, batch.done)

    def _get_action(self, env, o):
        epsilon = self._get_epsilon()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            self.model_learner.eval()
            model_out = self.model_learner(torch.tensor(o, device=self.device, dtype=torch.float).unsqueeze(0))
            action = model_out.max(1)[1].detach().to('cpu').numpy()[0]
        o_, reward, done, info = env.step(action)
        self.elapsed_env_steps += 1
        return action, o_, reward, done, info

    def _is_gather_experience(self):
        return len(self.replay_buffer) < self.replay_buffer_min_experience

    def _get_epsilon(self):
        if self._is_gather_experience():
            return 1. # np.random.random() < 1 always; hence do random action until minimum experience
        epsilon = self.epsilon_scheduler.get_epsilon(self.elapsed_model_steps) if self.epsilon_scheduler_use_steps \
            else self.epsilon_scheduler.get_epsilon(self.elapsed_episodes)
        return epsilon

    def _eval(self, env, epsilon=0.):
        returns = []
        for ep in range(100): #TODO change hardcoded value
            o = env.reset()
            done = False
            ret = 0
            while not done:
                self.model_target.eval()
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.model_target(torch.tensor(o, device=self.device, dtype=torch.float).unsqueeze(0)).\
                        max(1)[1].to('cpu').numpy()[0]
                o, rew, done, info = env.step(action)
                ret += rew
            returns.append(ret)
        print('mean eval return:', np.mean(returns))
        print()
