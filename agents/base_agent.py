import torch.nn as nn
import torch.optim as optim
from utils.scheduler.constant_scheduler import StepDecayScheduler


class BaseAgent:

    def __init__(self,  model_class, model_params, rng, device='cpu', n_episodes=2000, lr=1e-3, momentum=0.9,
                 criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99, epsilon_scheduler=StepDecayScheduler(),
                 epsilon_scheduler_use_steps=True, target_update_frequency=1e4, parameter_update_frequency=1, grad_clamp=None):

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
