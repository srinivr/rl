import torch.nn as nn
import torch.optim as optim
from utils.scheduler.constant_scheduler import StepDecayScheduler
import numpy as np
import torch


class BaseAgent:

    def __init__(self,  model_class, model_params, rng, device='cpu', n_episodes=2000, lr=1e-3, momentum=0.9,
                 criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99, epsilon_scheduler=StepDecayScheduler(),
                 epsilon_scheduler_use_steps=True, target_update_frequency=1e4, parameter_update_frequency=1,
                 grad_clamp=None):

        self.model_class = model_class
        self.rng = rng
        self.device = device
        self.n_episodes = n_episodes
        self.lr = lr
        self.momentum = momentum
        self.criterion = criterion()
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_scheduler_use_steps = epsilon_scheduler_use_steps
        self.model_learner = self.model_class(*model_params)
        self.model_target = self.model_class(*model_params)
        self.target_update_frequency = target_update_frequency
        self.parameter_update_frequency = parameter_update_frequency
        self.grad_clamp = grad_clamp
        self.model_learner.to(self.device)
        self.model_target.to(self.device)
        self.optimizer = optimizer(self.model_learner.parameters(), lr=self.lr, momentum=self.momentum)
        self.model_target.load_state_dict(self.model_learner.state_dict())
        self.elapsed_steps = 0
        self.elapsed_episodes = 0

    def learn(self, env):
        raise NotImplementedError

    def _get_epsilon(self, arg=None):
        raise NotImplementedError

    def _outputs(self, states, actions):
        self.model_learner.train()
        return self.model_learner(states).gather(1, actions.view(actions.size()[0], -1))

    def _action(self, env, o, env_idx=None):
        epsilon = self._get_epsilon(env_idx)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            self.model_learner.eval()
            action = self.model_learner(torch.tensor(o, device=self.device, dtype=torch.float).unsqueeze(0)) \
                .max(1)[1].detach().to('cpu').numpy()[0]
        o_, reward, done, info = env.step(action)
        return action, o_, reward, done, info

    def _step_updates(self, states, actions, targets):
        outputs = self._outputs(states, actions)
        loss = self.criterion(outputs, targets.view(targets.size()[0], -1))
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clamp:
            for p in self.model_learner.parameters():
                p.grad.data.clamp(*self.grad_clamp)
        self.optimizer.step()
        self.elapsed_steps += 1
        if self.elapsed_steps % self.target_update_frequency == 0:
            self.model_target.load_state_dict(self.model_learner.state_dict())
        if self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _episode_updates(self):
        self.elapsed_episodes += 1
        if not self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()
