import itertools
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
from utils.scheduler.decay_scheduler import DecayScheduler
import numpy as np


class NStepSynchronousDQNAgent(BaseAgent):
    """
    https://arxiv.org/pdf/1710.11417.pdf (batched n-step)
    """

    def __init__(self, model_class, model_params, rng, device='cpu', training_evaluation_frequency=100, optimizer=optim.RMSprop,
                 optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, criterion=nn.SmoothL1Loss, gamma=0.99,
                 epsilon_scheduler=DecayScheduler(decay=0.999995), target_synchronize_steps=1e4,
                 parameter_update_frequency=1, grad_clamp=None, n_step=5, n_envs=1):

        self.n_step = n_step
        self.n_envs = n_envs
        target_synchronize_steps = max(1, int(target_synchronize_steps // self.n_step))  # model is updated every n_step hence divide by n_step
        self.batch_values = namedtuple('Values', 'done step_ctr rewards states actions targets')
        super().__init__(model_class, model_params, rng, device, None, training_evaluation_frequency, optimizer,
                         optimizer_parameters, criterion, gamma, epsilon_scheduler, True, target_synchronize_steps,
                         parameter_update_frequency, grad_clamp)

    def learn(self, envs, eval_env):
        """
        env and eval_env should be different! (since we are using SubProcvecEnv and _eval calls env.reset())
        """
        assert eval_env is not None
        max_steps = 10000000  # TODO change hardcoded value
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, = [], [], [], [], []
        step_states = envs.reset()
        while self.elapsed_env_steps < max_steps:
            step_actions, step_next_states, step_rewards, step_done, step_info = self._get_epsilon_greedy_action(envs,
                                                                                                                 step_states)
            for b, s in zip([batch_states, batch_actions, batch_next_states, batch_rewards, batch_done],
                            [step_states, step_actions, step_next_states, step_rewards, step_done]):
                b.append(s)
            if self.elapsed_env_steps % self.n_step == 0:
                states, actions, targets = self._get_batch(batch_states, batch_actions, batch_next_states,
                                                           batch_rewards,
                                                           batch_done)  # batched n-step targets
                self._step_updates(states, actions, targets)  # TODO model steps updated here!!! So much challenges
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_done = [], [], [], [], []
            step_states = step_next_states
            if self.elapsed_env_steps % self.training_evaluation_frequency == 0:  # TODO naming: eval_steps or eval_episodes?
                print('step:', self.elapsed_env_steps, end=' ')
                self._eval(eval_env)

    def _get_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done):
        """
        construct n_step targets using super()._get_batch()
        """
        states, actions, targets = [], [], []
        _targets = None
        for i in range(1, self.n_step + 1):
            _states, _actions, _targets = super()._get_batch(batch_states[-i], batch_actions[-i], batch_next_states[-i],
                                                             batch_rewards[-i], batch_done[-i], future_target=_targets)
            states.insert(0, _states), actions.insert(0, _actions), targets.insert(0, _targets)
        return torch.cat(states), torch.cat(actions), torch.cat(targets)

    def _get_sample_action(self, envs):
        return [envs.action_space.sample() for _ in range(self.n_envs)]

    def _get_action_from_model(self, model, o, action_type='list'):
        return super()._get_action_from_model(model, o, action_type)

