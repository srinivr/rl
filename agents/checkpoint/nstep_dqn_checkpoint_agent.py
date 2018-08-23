import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.scheduler.linear_scheduler import LinearScheduler
from agents.nstep_dqn_agent import NStepSynchronousDQNAgent


class NStepDQNCheckpointAgent(NStepSynchronousDQNAgent):

    def __init__(self, model_class, model_params, rng, device='cpu', max_steps=1000000000,
                 training_evaluation_frequency=10000,
                 optimizer=optim.RMSprop, optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, lr_scheduler_fn=None,
                 criterion=nn.SmoothL1Loss, gamma=0.99, epsilon_scheduler=LinearScheduler(decay_steps=5e4),
                 target_synchronize_steps=1e4, td_losses=None, grad_clamp=None, grad_clamp_parameters=None, n_step=5,
                 n_processes=1, auxiliary_losses=None, input_transforms=None, output_transforms=None,
                 auxiliary_env_info=None, checkpoint_epsilon_frequency=None,
                 checkpoint_warmup_steps=None, checkpoint_epsilon_scheduler_template=None, log=True, log_dir=None,
                 save_checkpoint=True):
        super().__init__(model_class, model_params, rng, device, max_steps, training_evaluation_frequency, optimizer,
                         optimizer_parameters, lr_scheduler_fn, criterion, gamma, epsilon_scheduler,
                         target_synchronize_steps, td_losses, grad_clamp, grad_clamp_parameters, n_step, n_processes,
                         auxiliary_losses, input_transforms, output_transforms, auxiliary_env_info, log, log_dir,
                         save_checkpoint)

        assert checkpoint_epsilon_frequency is not None
        assert checkpoint_epsilon_scheduler_template is not None
        self.checkpoint_warmup_steps = 0 if checkpoint_warmup_steps is None else checkpoint_warmup_steps
        self.checkpoint_epsilon_scheduler_template = checkpoint_epsilon_scheduler_template
        self.checkpoint_frequency = checkpoint_epsilon_frequency
        self.checkpoint_values = [
            float('inf')]  # [-1] is always inf; threshold to use next scheduler
        self.epsilon_schedulers = [copy.deepcopy(self.checkpoint_epsilon_scheduler_template)]
        self.n_avg_episodes = 50  # number of episodes to average over to compute checkpoint
        self.reset_value = self.current_reset_value = checkpoint_epsilon_scheduler_template.get_final_epsilon()
        self.epsilon_reset_decay = 0.9
        self.epsilon_scheduler_index = np.zeros(self.n_processes, dtype=np.int64)

    def _get_sample_action(self, envs):
        if self.elapsed_env_steps < self.checkpoint_warmup_steps:
            return [envs.action_space.sample() for _ in range(self.n_processes)]
        else:
            raise NotImplementedError

    def _update_epsilon_scheduler(self, episode_returns, step_done, epsilon_scheduler_index):
        np_done = np.array(step_done)
        for idx in range(self.n_processes):
            if episode_returns[idx] > self.checkpoint_values[epsilon_scheduler_index[idx]]:
                epsilon_scheduler_index[idx] += 1
        epsilon_scheduler_index[np_done] = 0

    def _get_epsilon_greedy_action(self, env, states):
        if self.elapsed_env_steps < self.checkpoint_warmup_steps:
            return super()._get_epsilon_greedy_action(env, states)
        else:
            # TODO make it faster here
            epsilon_scheduler_index = self.epsilon_scheduler_index
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

    def _do_after_env_step(self, episode_returns, episode_lengths, step_rewards, step_done):
        super()._do_after_env_step(episode_returns, episode_lengths, step_rewards, step_done)
        if self.elapsed_env_steps > self.checkpoint_warmup_steps:
            self._update_epsilon_scheduler(episode_returns, step_done, self.epsilon_scheduler_index)

    def _do_after_iteration(self, cumulative_returns):
        if self.elapsed_env_steps > self.checkpoint_warmup_steps and self.elapsed_env_steps % \
                self.checkpoint_frequency == 0:
            if len(cumulative_returns) >= self.n_avg_episodes:
                # We can't have negative checkpoint values
                _temp_checkpoint = max(0., np.mean(cumulative_returns[-self.n_avg_episodes:]))
            else:
                _temp_checkpoint = np.float('-inf')
            if len(self.checkpoint_values) == 1 or _temp_checkpoint > self.checkpoint_values[-2]:
                self.checkpoint_values.insert(-1, _temp_checkpoint)
                self.epsilon_schedulers.append(copy.deepcopy(self.checkpoint_epsilon_scheduler_template))
                if self.log:
                    self.writer.add_scalar('data/checkpoint', self.checkpoint_values[-2], self.elapsed_env_steps)
                if len(cumulative_returns) >= self.n_avg_episodes:  # clear out returns only if we used them
                    cumulative_returns.clear()  # TODO verify if clearing is necessary
                    self.current_reset_value = self.reset_value
            elif _temp_checkpoint < self.checkpoint_values[-2] and len(cumulative_returns) >= self.n_avg_episodes:
                # boost
                decay = 1.
                for epsilon_scheduler in self.epsilon_schedulers:
                    epsilon_scheduler.reset(min(1., epsilon_scheduler.get_epsilon() +
                                                max(0.01, self.current_reset_value * decay)))
                    decay *= self.epsilon_reset_decay
                self.current_reset_value = min(1., self.current_reset_value + self.reset_value)

