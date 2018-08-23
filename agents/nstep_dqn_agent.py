import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from utils.scheduler.linear_scheduler import LinearScheduler


class NStepSynchronousDQNAgent(BaseAgent):
    """
    https://arxiv.org/pdf/1710.11417.pdf (batched, up-to) n-step
    """

    def __init__(self, model_class, model_params, rng, device='cpu', max_steps=1000000000,
                 training_evaluation_frequency=10000,
                 optimizer=optim.RMSprop, optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, lr_scheduler_fn=None,
                 criterion=nn.SmoothL1Loss, gamma=0.99, epsilon_scheduler=LinearScheduler(decay_steps=5e4),
                 target_synchronize_steps=1e4, td_losses=None, grad_clamp=None, grad_clamp_parameters=None, n_step=5,
                 n_processes=1, auxiliary_losses=None, input_transforms=None, output_transforms=None,
                 auxiliary_env_info=None, log=True, log_dir=None, save_checkpoint=True):

        self.max_steps = max_steps
        self.n_step = n_step
        self.n_processes = n_processes
        target_synchronize_steps = max(1, int(target_synchronize_steps // (
                self.n_step * self.n_processes)))  # model is updated every t_s_s environment steps; unit is model_steps
        super().__init__(model_class, model_params, rng, device, training_evaluation_frequency,
                         optimizer, optimizer_parameters, lr_scheduler_fn, criterion, gamma, epsilon_scheduler, True,
                         target_synchronize_steps, td_losses, grad_clamp, grad_clamp_parameters, auxiliary_losses,
                         input_transforms, output_transforms, auxiliary_env_info, log, log_dir, save_checkpoint)

    def learn(self, envs, eval_env=None, n_learn_steps=None, n_eval_episodes=100, step_states=None,
              episode_returns=None, episode_lengths=None, cumulative_returns=None):

        step_count, n_learn_steps, step_states, episode_returns, episode_lengths, cumulative_returns = self._setup_learn\
            (envs, eval_env, n_learn_steps, step_states, episode_returns, episode_lengths, cumulative_returns)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, batch_auxiliary_info = [], [], [], \
                                                                                                          [], [], []
        while step_count < n_learn_steps:
            step_count += 1
            step_actions, step_next_states, step_rewards, step_done, step_info, *auxiliary_info = \
                self._get_epsilon_greedy_action_and_step(envs, step_states)
            self._update_episode_values(episode_returns, episode_lengths, step_rewards, step_done,
                                        cumulative_returns)  # episode housekeeping
            self._do_after_env_step(episode_returns, episode_lengths, step_rewards, step_done)

            for b, s in zip([batch_states, batch_actions, batch_next_states, batch_rewards, batch_done],
                            [step_states, step_actions, step_next_states, step_rewards, step_done]):
                b.append(s)
            batch_auxiliary_info.extend(auxiliary_info)
            if self.elapsed_env_steps % self.n_step == 0:
                # if step_count % self.n_step == 0:  # TODO Mismatch in n step -> n_learn not multiple of batch_size
                self._step_updates(*self._get_nstep_batch(batch_states, batch_actions, batch_next_states, batch_rewards,
                                                          batch_done), self._get_auxiliary_batch(*batch_auxiliary_info))
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, batch_auxiliary_info = \
                    [], [], [], [], [], []
            self._do_after_iteration(cumulative_returns)
            step_states = step_next_states
            self._reset_episode_values(episode_returns, episode_lengths, step_done)
            self._do_eval(eval_env, n_eval_episodes)
        return step_states, episode_returns, episode_lengths, cumulative_returns

    def _get_nstep_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done):
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

    @staticmethod
    def _update_episode_values(episode_returns, episode_lengths, step_rewards, step_done, cumulative_returns):
        episode_returns += step_rewards
        episode_lengths += 1
        np_done = np.array(step_done)
        if np.sum(np_done) != 0:
            cumulative_returns.extend(episode_returns[np_done])

    @staticmethod
    def _reset_episode_values(episode_returns, episode_lengths, step_done):
        np_done = np.array(step_done)
        episode_returns[np_done] = 0.
        episode_lengths[np_done] = 0.

    def _do_after_env_step(self, episode_returns, episode_lengths, step_rewards, step_done):
        if self.log:
            self._training_log(episode_returns, episode_lengths, step_done)

    def _do_eval(self, env, n_episodes):
        if env and self.elapsed_env_steps % self.training_evaluation_frequency == 0:
            print('step:', self.elapsed_env_steps, end=' ')
            self._eval(env, n_episodes=n_episodes, epsilon=self.epsilon_scheduler.get_final_epsilon())

    def _setup_learn(self, envs, eval_env, n_learn_steps, step_states, episode_returns, episode_lengths,
                     cumulative_returns):
        """
        set empty values if not restarting else return appropriate input values
        """
        if not eval_env:
            print('no evaluation environment specified. No results will be printed!!')
        if n_learn_steps is None:
            n_learn_steps = self.max_steps
        assert self.n_step * self.n_processes <= n_learn_steps <= self.max_steps  # at least 1 batch
        assert n_learn_steps % (self.n_step * self.n_processes) == 0
        step_count = 0
        n_learn_steps = n_learn_steps // self.n_processes  # to keep counting simple
        if step_states is None:  # if not None => restarting from given state
            step_states = envs.reset()
            step_states = self._apply_input_transform(step_states)
            episode_returns, episode_lengths = np.zeros(self.n_processes), np.zeros(self.n_processes)
            cumulative_returns = []
        return step_count, n_learn_steps, step_states, episode_returns, episode_lengths, cumulative_returns

    def _training_log(self, episode_returns, episode_lengths, step_done):
        if self.log:
            np_done = np.array(step_done)
            if np.sum(np_done) != 0:
                self.writer.add_scalar('data/train_rewards', np.sum(episode_returns[np_done]) / np.sum(step_done),
                                       self.elapsed_env_steps)
                self.writer.add_scalar('data/train_episode_length',
                                       np.sum(episode_lengths[np_done]) / np.sum(step_done),
                                       self.elapsed_env_steps)
