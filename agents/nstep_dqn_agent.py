import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple
from agents.base_agent import BaseAgent
from utils.scheduler.linear_scheduler import LinearScheduler


# TODO make scheduler steps consistent
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
                 checkpoint_epsilon=False, checkpoint_epsilon_frequency=None, auxiliary_env_info=None, log=True,
                 log_dir=None):

        self.max_steps = max_steps
        self.n_step = n_step
        self.n_processes = n_processes
        target_synchronize_steps = max(1, int(target_synchronize_steps // (
                self.n_step * self.n_processes)))  # model is updated every t_s_s environment steps
        self.batch_values = namedtuple('Values', 'done step_ctr rewards states actions targets')
        self.checkpoint_epsilon = checkpoint_epsilon
        super().__init__(model_class, model_params, rng, device, training_evaluation_frequency,
                         optimizer, optimizer_parameters, lr_scheduler_fn, criterion, gamma, epsilon_scheduler, True,
                         target_synchronize_steps, td_losses, grad_clamp, grad_clamp_parameters, auxiliary_losses,
                         input_transforms, output_transforms, auxiliary_env_info, log, log_dir)

        if self.checkpoint_epsilon:
            assert checkpoint_epsilon_frequency is not None
            self.checkpoint_frequency = checkpoint_epsilon_frequency
            self.checkpoint_values = [
                float('inf')]  # [-1] is always inf; threshold to use next scheduler
            self.original_epsilon_scheduler = copy.deepcopy(self.epsilon_scheduler)  # template to create new schedulers
            self.epsilon_schedulers = [copy.deepcopy(self.original_epsilon_scheduler)]

    def learn(self, envs, eval_env=None, n_learn_iterations=None, n_eval_episodes=100, step_states=None,
              episode_returns=None, episode_lengths=None):
        """
        env and eval_env should be different! (since we are using SubProcvecEnv and _eval calls env.reset())
        """
        if not eval_env:
            print('no evaluation environment specified. No results will be printed!!')
        if n_learn_iterations is None:
            n_learn_iterations = self.max_steps
        assert self.n_step * self.n_processes <= n_learn_iterations <= self.max_steps  # steps to do at least 1 batch
        n_learn_iterations = n_learn_iterations // self.n_processes  # to keep counting simple  # TODO ensure mod == 0
        ephemeral_step_count = 0

        batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, batch_auxiliary_info = [], [], [], \
                                                                                                          [], [], []

        if step_states is None:  # if not None => restarting from given state
            step_states = envs.reset()
            step_states = self._apply_input_transform(step_states)
            episode_returns, episode_lengths = np.zeros(self.n_processes), np.zeros(self.n_processes)  # training eval

        cumulative_returns = []
        # epsilon scheduler
        epsilon_scheduler_index = np.zeros(self.n_processes, dtype=np.int64) if self.checkpoint_epsilon else None

        while ephemeral_step_count < n_learn_iterations:
            ephemeral_step_count += 1
            step_actions, step_next_states, step_rewards, step_done, step_info, *auxiliary_info = \
                self._get_epsilon_greedy_action_and_step(envs, step_states, epsilon_scheduler_index)
            self._update_episode_values(episode_returns, episode_lengths, step_rewards, step_done,
                                        cumulative_returns)  # episode housekeeping
            if self.checkpoint_epsilon:
                self._update_epsilon_scheduler(episode_returns, step_done, epsilon_scheduler_index)
            if self.log:
                self._training_log(episode_returns, episode_lengths, step_done)

            for b, s in zip([batch_states, batch_actions, batch_next_states, batch_rewards, batch_done],
                            [step_states, step_actions, step_next_states, step_rewards, step_done]):
                b.append(s)
            batch_auxiliary_info.extend(auxiliary_info)
            if self.elapsed_env_steps % self.n_step == 0:
                # if ephemeral_step_count % self.n_step == 0:  # TODO Mismatch in n step -> n_learn not multiple of batch_size
                states, actions, rewards, targets, batch_done = self.__get_batch(batch_states, batch_actions,
                                                                                 batch_next_states,
                                                                                 batch_rewards,
                                                                                 batch_done)  # batched n-step targets
                #  notice above batch_done has been from list of lists to a list
                batch_auxiliary_info = self._get_auxiliary_batch(*batch_auxiliary_info)
                self._step_updates(states, actions, rewards, targets, batch_done, batch_auxiliary_info)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, batch_auxiliary_info = \
                    [], [], [], [], [], []
            step_states = step_next_states
            if self.checkpoint_epsilon and self.elapsed_env_steps % self.checkpoint_frequency == 0:
                # TODO notice 1.2 below
                # if len(self.checkpoint_values) == 1 or np.mean(cumulative_returns) > 1.2 * self.checkpoint_values[-2]:
                if len(self.checkpoint_values) == 1 or np.mean(cumulative_returns) > self.checkpoint_values[-2]:
                    self.checkpoint_values.insert(-1, np.mean(cumulative_returns))
                    self.epsilon_schedulers.append(copy.deepcopy(self.original_epsilon_scheduler))
                    if self.log:
                        self.writer.add_scalar('data/checkpoint', self.checkpoint_values[-2], self.elapsed_env_steps)
                cumulative_returns = []
            self._reset_episode_values(episode_returns, episode_lengths, step_done)
            # beyond this point all episode variable must have been reset
            if eval_env and self.elapsed_env_steps % self.training_evaluation_frequency == 0:
                print('step:', self.elapsed_env_steps, end=' ')
                self._eval(eval_env, n_eval_episodes)

        return step_states, episode_returns, episode_lengths

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
        if not self.checkpoint_epsilon:
            return [envs.action_space.sample() for _ in range(self.n_processes)]
        else:
            raise NotImplementedError

    def _get_greedy_action(self, model, state, action_type='list'):
        return super()._get_greedy_action(model, state, action_type)

    def _get_n_steps(self):
        return self.n_processes

    def _update_epsilon_scheduler(self, episode_returns, step_done, epsilon_scheduler_index):
        np_done = np.array(step_done)
        for idx in range(self.n_processes):
            # TODO notice the change below: cross threshold if the return is up to 30% worse of checkpoint
            # if episode_returns[idx] > self.checkpoint_values[epsilon_scheduler_index[idx]] - \
            #         np.abs(self.checkpoint_values[epsilon_scheduler_index[idx]] * 0.3):
            if episode_returns[idx] > self.checkpoint_values[epsilon_scheduler_index[idx]]:
                epsilon_scheduler_index[idx] += 1
        epsilon_scheduler_index[np_done] = 0

    def _update_episode_values(self, episode_returns, episode_lengths, step_rewards, step_done, cumulative_returns):
        episode_returns += step_rewards
        episode_lengths += 1
        np_done = np.array(step_done)
        if np.sum(np_done) != 0:
            cumulative_returns.extend(episode_returns[np_done])

    def _reset_episode_values(self, episode_returns, episode_lengths, step_done):
        np_done = np.array(step_done)
        episode_returns[np_done] = 0.
        episode_lengths[np_done] = 0.

    def _training_log(self, episode_returns, episode_lengths, step_done):
        if self.log:
            np_done = np.array(step_done)
            if np.sum(np_done) != 0:
                self.writer.add_scalar('data/train_rewards', np.sum(episode_returns[np_done]) / np.sum(step_done),
                                       self.elapsed_env_steps)
                self.writer.add_scalar('data/train_episode_length', np.sum(episode_lengths[np_done]) / np.sum(step_done),
                                       self.elapsed_env_steps)

    def _get_epsilon_greedy_action(self, env, states, *args):
        if not self.checkpoint_epsilon:
            return super()._get_epsilon_greedy_action(env, states, args)
        else:
            # TODO make it faster here
            epsilon_scheduler_index = args[0][0]
            actions = []

            self.model_learner.eval()
            greedy_actions = self._get_greedy_action(self.model_learner, states)

            for idx in range(self.n_processes):
                if np.random.random() < self.epsilon_schedulers[epsilon_scheduler_index[idx]].get_epsilon():
                    actions.append(env.action_space.sample())
                else:
                    actions.append(greedy_actions[idx])
                self.epsilon_schedulers[epsilon_scheduler_index[idx]].step()
                if self.log:
                    self.writer.add_scalar('data/epsilon_dynamic', self.epsilon_schedulers[epsilon_scheduler_index[idx]]
                                           .get_epsilon(), self.elapsed_env_steps + idx + 1)
            return actions
