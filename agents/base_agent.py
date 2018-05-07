import torch.nn as nn
import torch.optim as optim
from utils.scheduler.decay_scheduler import StepDecayScheduler
import numpy as np
import torch


class BaseAgent:

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000, n_eval_steps=100, lr=1e-3,
                 momentum=0.9, criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99,
                 epsilon_scheduler=StepDecayScheduler(), epsilon_scheduler_use_steps=True, target_update_steps=1e4,
                 parameter_update_frequency=1, grad_clamp=None):

        self.model_class = model_class
        self.rng = rng
        self.device = device
        self.n_episodes = n_episodes
        self.n_eval_steps = n_eval_steps
        self.lr = lr
        self.momentum = momentum
        self.criterion = criterion()
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_scheduler_use_steps = epsilon_scheduler_use_steps
        self.model_learner = self.model_class(*model_params)
        self.model_target = self.model_class(*model_params)
        self.target_update_steps = target_update_steps
        self.parameter_update_frequency = parameter_update_frequency
        self.grad_clamp = grad_clamp
        self.model_learner.to(self.device)
        self.model_target.to(self.device)
        self.optimizer = optimizer(self.model_learner.parameters(), lr=self.lr, momentum=self.momentum)
        self.model_target.load_state_dict(self.model_learner.state_dict())
        self.elapsed_model_steps = 0  # to synchronize models
        self.elapsed_env_steps = 0  # number of interactions with the environment
        self.elapsed_episodes = 0

    def learn(self, env):
        raise NotImplementedError

    def _get_epsilon(self, *args):
        raise NotImplementedError

    def _get_action(self, *args):
        raise NotImplementedError

    def _step_updates(self, states, actions, targets):
        """

        :param states: pytorch float tensor
        :param actions: pytorch long tensor
        :param targets: pytorch float tensor
        :return:
        """
        if states.size()[0] < 2:
            print('Batch has only 1 example. Can cause problems if batch norm was used.. Skipping step')
            return
        self.model_learner.train()
        outputs = self.model_learner(states).gather(1, actions.view(actions.size()[0], -1))
        loss = self.criterion(outputs, targets.view(targets.size()[0], -1))
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clamp:
            for p in self.model_learner.parameters():
                p.grad.data.clamp(*self.grad_clamp)
        self.optimizer.step()
        self.elapsed_model_steps += 1
        if self.elapsed_model_steps % self.target_update_steps == 0:
            self.model_target.load_state_dict(self.model_learner.state_dict())
            print('agents synchronized...')
        if self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _episode_updates(self):
        self.elapsed_episodes += 1
        if not self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _get_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, future_target=None):
        """

        :param batch_states:
        :param batch_actions:
        :param batch_next_states:
        :param batch_rewards:
        :param batch_done:
        :param future_target:
        :return: torch tensors
        """
        float_args = dict(device=self.device, dtype=torch.float)
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch_done)), device=self.device)
        states = torch.tensor(batch_states, **float_args)
        actions = torch.tensor(batch_actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(batch_rewards, **float_args)
        targets = torch.zeros(len(actions), device=self.device)
        if future_target:
            targets[non_final_mask] += self.gamma * future_target[non_final_mask]
        elif not all(batch_done):
            _next_non_final_states = torch.tensor(batch_next_states, **float_args)[non_final_mask]
            #next_non_final_states = torch.tensor([s for s, d in zip(batch_next_states, batch_done) if not d], **float_args)
            #print('_nnfs', len(_next_non_final_states), 'nnfs:', len(next_non_final_states), torch.sum(_next_non_final_states - next_non_final_states))
            self.model_target.eval()
            targets[non_final_mask] += self.gamma * self.model_target(_next_non_final_states).max(1)[0].detach()
        targets += rewards
        return states, actions, targets

    def _eval(self):
        raise NotImplementedError
