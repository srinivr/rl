import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple
from agents.base_agent import BaseAgent
from utils.scheduler.linear_scheduler import LinearScheduler


class NStepSynchronousDQNAgent(BaseAgent):
    """
    https://arxiv.org/pdf/1710.11417.pdf (batched, up-to) n-step
    """

    def __init__(self, experiment_id, model_class, model_params, rng, device='cpu', max_steps=1000000000,
                 training_evaluation_frequency=10000,
                 optimizer=optim.RMSprop, optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, criterion=nn.SmoothL1Loss,
                 gamma=0.99, epsilon_scheduler=LinearScheduler(decay_steps=5e4), target_synchronize_steps=1e4,
                 td_losses=None, grad_clamp=None, n_step=5, n_processes=1, auxiliary_losses=None, input_transforms=None,
                 output_transforms=None, log=True):

        self.max_steps = max_steps
        self.n_step = n_step
        self.n_processes = n_processes
        target_synchronize_steps = max(1, int(target_synchronize_steps // (
                self.n_step * self.n_processes)))  # model is updated every t_s_s environment steps
        self.batch_values = namedtuple('Values', 'done step_ctr rewards states actions targets')
        super().__init__(experiment_id, model_class, model_params, rng, device, training_evaluation_frequency,
                         optimizer,
                         optimizer_parameters, criterion, gamma, epsilon_scheduler, True, target_synchronize_steps,
                         td_losses, grad_clamp, auxiliary_losses, input_transforms, output_transforms, log)

    def learn(self, envs, eval_env=None, n_learn_iterations=None, n_eval_steps=100, step_states=None,
              episode_rewards=None, episode_lengths=None):
        """
        env and eval_env should be different! (since we are using SubProcvecEnv and _eval calls env.reset())
        """
        # assert eval_env is not None
        if not eval_env:
            print('no evaluation environment specified. No results will be printed!!')
        if n_learn_iterations is None:
            n_learn_iterations = self.max_steps
        assert self.n_step * self.n_processes <= n_learn_iterations <= self.max_steps  # steps to do at least 1 batch
        n_learn_iterations = n_learn_iterations // self.n_processes  # to keep counting simple  # TODO ensure mod == 0
        ephemeral_step_count = 0

        batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, = [], [], [], [], []

        if step_states is None:  # if not None => restarting from given state
            step_states = envs.reset()
            step_states = self._apply_input_transform(step_states)
            episode_rewards, episode_lengths = np.zeros(self.n_processes), np.zeros(self.n_processes)  # training eval

        while ephemeral_step_count < n_learn_iterations:
            ephemeral_step_count += 1
            step_actions, step_next_states, step_rewards, step_done, step_info = self._get_epsilon_greedy_action_and_step(
                envs,
                step_states)
            if self.log:
                self._training_log(episode_rewards, episode_lengths, step_rewards, step_done)

            for b, s in zip([batch_states, batch_actions, batch_next_states, batch_rewards, batch_done],
                            [step_states, step_actions, step_next_states, step_rewards, step_done]):
                b.append(s)
            if self.elapsed_env_steps % self.n_step == 0:
                # if ephemeral_step_count % self.n_step == 0:  # TODO Mismatch in n step -> n_learn not multiple of batch_size
                states, actions, rewards, targets, batch_done = self.__get_batch(batch_states, batch_actions,
                                                                                 batch_next_states,
                                                                                 batch_rewards,
                                                                                 batch_done)  # batched n-step targets
                #  notice above batch_done has been from list of lists to a list
                self._step_updates(states, actions, rewards, targets, batch_done)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_done = [], [], [], [], []
            step_states = step_next_states
            if eval_env and self.elapsed_env_steps % self.training_evaluation_frequency == 0:
                print('step:', self.elapsed_env_steps, end=' ')
                self._eval(eval_env, n_eval_steps)
        return step_states, episode_rewards, episode_lengths

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

    def _training_log(self, episode_rewards, episode_lengths, step_rewards, step_done):
        episode_rewards += step_rewards
        episode_lengths += 1
        np_done = np.array(step_done)
        if np.sum(np_done) != 0:
            self.writer.add_scalar('data/train_rewards', np.sum(episode_rewards[np_done]) / np.sum(step_done),
                                   self.elapsed_env_steps)
            self.writer.add_scalar('data/train_episode_length', np.sum(episode_lengths[np_done]) / np.sum(step_done),
                                   self.elapsed_env_steps)
            episode_rewards[np_done] = 0.
            episode_lengths[np_done] = 0.
