import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.scheduler.decay_scheduler import DecayScheduler
from collections import namedtuple
import numpy as np


class DQNAgent(BaseAgent):

    def __init__(self, experiment_id, model_class, model_params, rng, device='cpu', n_episodes=2000,
                 training_evaluation_frequency=100, optimizer=optim.RMSprop, optimizer_parameters=
                 {'lr': 1e-3, 'momentum': 0.9}, criterion=nn.SmoothL1Loss, gamma=0.99, epsilon_scheduler=
                 DecayScheduler(), epsilon_scheduler_use_steps=True, target_synchronize_steps=1e4,
                 parameter_update_steps=1, grad_clamp=None, mb_size=32, replay_buffer_size=100000,
                 replay_buffer_min_experience=None, auxiliary_losses=None):

        self.n_episodes = n_episodes
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
        super().__init__(experiment_id, model_class, model_params, rng, device, training_evaluation_frequency, optimizer,
                         optimizer_parameters, criterion, gamma, epsilon_scheduler, epsilon_scheduler_use_steps,
                         target_synchronize_steps, parameter_update_steps, grad_clamp, auxiliary_losses)

    def learn(self, env, eval_env=None, n_eval_episodes=100):
        if not eval_env:
        #     eval_env = env
            print('no evaluation environment specified. evaluation will not be performed..')
        returns = []
        # self.epsilon_scheduler.set_no_decay()  # do not decay until replay buffer is adequately filled
        for ep in range(self.n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                action, o_, reward, done, info = self._get_epsilon_greedy_action(env, o)
                ret += reward
                self.replay_buffer.insert(self.transitions(o, action, reward, o_, done))
                if self._is_gather_experience():
                    continue
                # self.epsilon_scheduler.set_do_decay()  # decay since replay buffer is adequately filled
                states, actions, rewards, targets, batch_done = self.__get_batch()  # note: __ not _
                self._step_updates(states, actions, rewards, targets, batch_done)
                o = o_
            self._episode_updates()
            returns.append(ret)
            if (ep+1) % self.training_evaluation_frequency == 0:
                print('mean training return', self.training_evaluation_frequency, ' returns:', ep, ':', np.mean(returns))
                if eval_env:
                    print('ep:', ep, end=' ')
                    self._eval(eval_env, n_eval_episodes)
                returns = []

    def __get_batch(self):
        samples = self.replay_buffer.sample(self.mb_size)
        batch = self.transitions(*zip(*samples))  # https://stackoverflow.com/a/19343/3343043
        return super()._get_batch(batch.state, batch.action, batch.next_state, batch.reward, batch.done)

    def _is_gather_experience(self):
        return len(self.replay_buffer) < self.replay_buffer_min_experience

    def _get_epsilon(self):
        if self._is_gather_experience():
            return 1.  # np.random.random() < 1 always; hence do random action until minimum experience
        return super()._get_epsilon()

    def _get_sample_action(self, env):
        return env.action_space.sample()

    def _get_action_from_model(self, model, state, action_type='scalar'):
        return super()._get_action_from_model(model, state, action_type)

