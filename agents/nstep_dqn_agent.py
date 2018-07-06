import itertools
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
from td_losses.q_loss import QLoss
from utils.scheduler.decay_scheduler import DecayScheduler


class NStepSynchronousDQNAgent(BaseAgent):
    """
    https://arxiv.org/pdf/1710.11417.pdf (batched, up-to) n-step
    """

    def __init__(self, experiment_id, model_class, model_params, rng, device='cpu', max_steps=1000000000,
                 training_evaluation_frequency=10000,
                 optimizer=optim.RMSprop, optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, criterion=nn.SmoothL1Loss,
                 gamma=0.99, epsilon_scheduler=DecayScheduler(decay=0.999995), target_synchronize_steps=1e4,
                 parameter_update_steps=1, grad_clamp=None, n_step=5, n_processes=1, auxiliary_losses=None):

        self.max_steps = max_steps
        self.n_step = n_step
        self.n_processes = n_processes
        target_synchronize_steps = max(1, int(target_synchronize_steps // (
                    self.n_step * self.n_processes)))  # model is updated every t_s_s environment steps
        self.batch_values = namedtuple('Values', 'done step_ctr rewards states actions targets')
        super().__init__(experiment_id, model_class, model_params, rng, device, training_evaluation_frequency, optimizer,
                         optimizer_parameters, criterion, gamma, epsilon_scheduler, True, target_synchronize_steps,
                         parameter_update_steps, grad_clamp, auxiliary_losses)
        self.td_losses.append(QLoss())

    def learn(self, envs, eval_env=None, n_eval_steps=100):
        """
        env and eval_env should be different! (since we are using SubProcvecEnv and _eval calls env.reset())
        """
        # assert eval_env is not None
        if not eval_env:
            print('no evaluation environment specified. No results will be printed!!')
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, = [], [], [], [], []
        step_states = envs.reset()
        episode_rewards = np.zeros(self.n_processes)  # Training evaluation
        while self.elapsed_env_steps < self.max_steps:
            step_actions, step_next_states, step_rewards, step_done, step_info = self._get_epsilon_greedy_action_and_step(envs,
                                                                                                                          step_states)
            if self.log:
                self._training_eval(episode_rewards, step_rewards, step_done)

            for b, s in zip([batch_states, batch_actions, batch_next_states, batch_rewards, batch_done],
                            [step_states, step_actions, step_next_states, step_rewards, step_done]):
                b.append(s)
            if self.elapsed_env_steps % self.n_step == 0:
                states, actions, rewards, targets, batch_done = self.__get_batch(batch_states, batch_actions,
                                                                                        batch_next_states,
                                                                                        batch_rewards,  batch_done)  # batched n-step targets
                #  notice above batch_done has been from list of lists to a list
                self._step_updates(states, actions, rewards, targets, batch_done)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_done = [], [], [], [], []
            step_states = step_next_states
            if eval_env and self.elapsed_env_steps % self.training_evaluation_frequency == 0:
                print('step:', self.elapsed_env_steps, end=' ')
                self._eval(eval_env, n_eval_steps)

    def __get_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done):
        """
        construct n_step targets using super()._get_batch()
        """
        states, actions, rewards, targets = [], [], [], []  # targets: list of list containing tensors
        _targets = None
        for i in range(1, self.n_step + 1):
            _states, _actions, _rewards, _targets, _ = super()._get_batch(batch_states[-i], batch_actions[-i],
                                                                       batch_next_states[-i], batch_rewards[-i],
                                                                       batch_done[-i], future_targets=_targets)
            states.insert(0, _states), actions.insert(0, _actions), rewards.insert(0, _rewards), targets.insert(0,
                                                                                                                _targets)
        batch_dones = list(itertools.chain(*batch_done))
        targets = zip(*targets)  # pay attention at this line
        return torch.cat(states), torch.cat(actions), torch.cat(rewards), [torch.cat(t) for t in targets], batch_dones

    def _get_sample_action(self, envs):
        return [envs.action_space.sample() for _ in range(self.n_processes)]

    def _get_greedy_action(self, model, state, action_type='list'):
        return super()._get_greedy_action(model, state, action_type)

    def _get_n_steps(self):
        return self.n_processes

    def _training_eval(self, episode_rewards, step_rewards, step_done):
        episode_rewards += step_rewards
        np_done = np.array(step_done)
        if np.sum(np_done) != 0:
            self.writer.add_scalar('data/train_rewards', np.sum(episode_rewards[np_done]) / np.sum(step_done),
                                   self.elapsed_env_steps)
            episode_rewards[np_done] = 0.
