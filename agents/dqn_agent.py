import copy

import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.scheduler.decay_scheduler import DecayScheduler
from collections import namedtuple
import numpy as np


class DQNAgent(BaseAgent):

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000,
                 training_evaluation_frequency=100, optimizer=optim.RMSprop, optimizer_parameters=
                 {'lr': 1e-3, 'momentum': 0.9}, lr_scheduler_fn=None, criterion=nn.SmoothL1Loss, gamma=0.99,
                 epsilon_scheduler=DecayScheduler(), epsilon_scheduler_use_steps=True, target_synchronize_steps=1e4,
                 td_losses=None, grad_clamp=None, grad_clamp_parameters=None, mb_size=32, replay_buffer_size=100000,
                 replay_buffer_min_experience=None, auxiliary_losses=None, input_transforms=[], output_transforms=[],
                 checkpoint_epsilon=False, checkpoint_epsilon_frequency=None, checkpoint_warmup_steps=None,
                 checkpoint_epsilon_scheduler_template=None, auxiliary_env_info=None, log=True, log_dir=None,
                 save_checkpoint=True):

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
        self.checkpoint_epsilon = checkpoint_epsilon
        super().__init__(model_class, model_params, rng, device, training_evaluation_frequency,
                         optimizer, optimizer_parameters, lr_scheduler_fn, criterion, gamma, epsilon_scheduler,
                         epsilon_scheduler_use_steps, target_synchronize_steps, td_losses, grad_clamp,
                         grad_clamp_parameters, auxiliary_losses, input_transforms, output_transforms,
                         auxiliary_env_info, log, log_dir, save_checkpoint)
        if self.checkpoint_epsilon:
            assert checkpoint_epsilon_frequency is not None
            assert checkpoint_epsilon_scheduler_template is not None
            self.checkpoint_warmup_steps = 0 if checkpoint_warmup_steps is None else checkpoint_warmup_steps
            self.checkpoint_epsilon_scheduler_template = checkpoint_epsilon_scheduler_template
            self.checkpoint_frequency = checkpoint_epsilon_frequency
            self.checkpoint_values = [float('inf')]  # [-1] is always infinity; threshold to use next scheduler
            self.epsilon_schedulers = [copy.deepcopy(self.checkpoint_epsilon_scheduler_template)]
            self.n_avg_episodes = 50  # number of episodes to average over to compute checkpoint

        if self.auxiliary_env_info:
            self.transitions = namedtuple('Transition', 'state action reward next_state done auxiliary')
        else:
            self.transitions = namedtuple('Transition', 'state action reward next_state done')

    def learn(self, env, eval_env=None, n_learn_iterations=None, n_eval_episodes=100):
        if not eval_env:
            print('no evaluation environment specified. evaluation will not be performed..')
        if n_learn_iterations is None:
            n_learn_iterations = self.n_episodes
        assert 1 <= n_learn_iterations <= self.n_episodes  # TODO learn + elapsed < n_episodes
        returns, episode_lengths = [], []
        ephemeral_episode_count = 0
        while ephemeral_episode_count < n_learn_iterations:
            ephemeral_episode_count += 1
            o = env.reset()
            o = self._apply_input_transform(o)
            done = False
            episode_return, episode_length = 0., 0.
            if self.checkpoint_epsilon:
                self.epsilon_scheduler, epsilon_scheduler_index = self.epsilon_schedulers[0], 0
            while not done:
                action, o_, reward, done, info, *auxiliary_info = self._get_epsilon_greedy_action_and_step(env, o)
                #if self.auxiliary_env_info:
                #    auxiliary_info = self.auxiliary_tuple(*auxiliary_info)
                episode_return += reward
                episode_length += 1
                self._insert_in_replay_buffer(o, action, reward, o_, done, *auxiliary_info)
                if self._is_gather_experience():
                    continue
                if self.checkpoint_epsilon:
                    if episode_return > self.checkpoint_values[epsilon_scheduler_index]:
                        epsilon_scheduler_index += 1
                        self.epsilon_scheduler = self.epsilon_schedulers[epsilon_scheduler_index]
                states, actions, rewards, targets, batch_done, batch_auxiliary_info = self.__get_batch()  # note: __ not _
                self._step_updates(states, actions, rewards, targets, batch_done, batch_auxiliary_info)
                o = o_
            self._episode_updates()
            if self.log:
                self._training_log(episode_return, episode_length)
            returns.append(episode_return)
            episode_lengths.append(episode_length)
            if self.checkpoint_epsilon and self.elapsed_episodes % self.checkpoint_frequency == 0:
                # TODO logic below will mess with boosting
                # Non-negative checkpoint
                _temp_checkpoint = max(0., np.mean(returns[-self.n_avg_episodes:]) if len(returns) >=
                                                                                      self.n_avg_episodes else np.float('-inf'))
                if len(self.checkpoint_values) == 1 or _temp_checkpoint > self.checkpoint_values[-2]:
                    self.checkpoint_values.insert(-1, _temp_checkpoint)
                    self.epsilon_schedulers.append(copy.deepcopy(self.checkpoint_epsilon_scheduler_template))
                    if self.log:
                        self.writer.add_scalar('data/checkpoint', self.checkpoint_values[-2], self.elapsed_env_steps)
                    if len(returns) >= self.n_avg_episodes:
                        returns, episode_lengths = [], []
            if eval_env and self.elapsed_episodes % self.training_evaluation_frequency == 0:
                print('ep:', self.elapsed_episodes, end=' ')
                self._eval(eval_env, n_episodes=n_eval_episodes, epsilon=0.05)

    def __get_batch(self):
        samples = self.replay_buffer.sample(self.mb_size)
        batch = self.transitions(*zip(*samples))  # https://stackoverflow.com/a/19343/3343043
        auxiliary_info = batch.auxiliary if self.auxiliary_env_info else []
        result = *super()._get_batch(batch.state, batch.action, batch.next_state, batch.reward, batch.done), self.\
            _get_auxiliary_batch(*auxiliary_info)
        return result

    def _is_gather_experience(self):
        return len(self.replay_buffer) < self.replay_buffer_min_experience

    def _get_epsilon(self):
        if self._is_gather_experience():
            return 1.  # np.random.random() < 1 always; hence do random action until minimum experience
        return super()._get_epsilon()

    def _get_sample_action(self, env):
        return env.action_space.sample()

    def _get_greedy_action(self, model, state, action_type='scalar'):
        return super()._get_greedy_action(model, state, action_type)

    def _get_n_steps(self):
        return 1

    def _insert_in_replay_buffer(self, state, action, reward, next_state, done, *auxiliary_info):
        if self.auxiliary_env_info:
            self.replay_buffer.insert(self.transitions(state, action, reward, next_state, done, auxiliary_info))
        else:
            self.replay_buffer.insert(self.transitions(state, action, reward, next_state, done))

    def _training_log(self, ret, length):
        if self.log:
            self.writer.add_scalar('data/train_rewards', ret, self.elapsed_env_steps)
            self.writer.add_scalar('data/train_episode_length', length, self.elapsed_env_steps)

    def _get_state(self):
        state = super()._get_state()
        if self.checkpoint_epsilon:
            state['checkpoint_values'] = self.checkpoint_values
            state['epsilon_schedulers'] = self.epsilon_schedulers
        return state

    def _set_state(self, state):
        if self.checkpoint_epsilon:
            self.checkpoint_values = state['checkpoint_values']
            self.epsilon_schedulers = state['epsilon_schedulers']
        super()._set_state(state)
