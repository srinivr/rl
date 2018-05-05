import torch
import torch.nn as nn
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer
from utils.scheduler.constant_scheduler import StepDecayScheduler
from collections import namedtuple
import numpy as np
import time


class DQNAgent:

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000, mb_size=32, lr=1e-3, momentum=0.9,
                 criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99, epsilon_scheduler=StepDecayScheduler(),
                 epsilon_scheduler_use_steps=True, target_update_frequency=1e4, parameter_update_frequency=1,
                 replay_buffer_size=100000, replay_build_wait_steps=None, grad_clamp=None):

        self.model_class = model_class
        self.rng = rng
        self.device = device
        self.n_episodes = n_episodes
        self.mb_size = mb_size
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
        self.replay_buffer_size = replay_buffer_size
        self.grad_clamp = grad_clamp
        if self.replay_buffer_size > 0:
            if replay_build_wait_steps:
                assert replay_build_wait_steps <= self.replay_buffer_size
                self.replay_build_wait_steps = replay_build_wait_steps
            else:
                self.replay_build_wait_steps = self.mb_size
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.model_learner.to(self.device)
        self.model_target.to(self.device)
        self.optimizer = optimizer(self.model_learner.parameters(), lr=self.lr, momentum=self.momentum)
        self.model_target.load_state_dict(self.model_learner.state_dict())
        self.elapsed_steps = 0
        self.elapsed_episodes = 0

    def _wait(self):
        return self.elapsed_steps <= self.replay_build_wait_steps

    def learn(self, env):
        """
        1 step
        :param env: gym like environment
        :return: None
        """
        Transition = namedtuple('Transition', 'state action reward next_state done')
        returns = []
        now = time.time()
        for ep in range(self.n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                epsilon = self.epsilon_scheduler.get_epsilon(self.elapsed_steps) if self.epsilon_scheduler_use_steps\
                        else self.epsilon_scheduler.get_epsilon(self.elapsed_episodes)
                if self._wait() or np.random.random() < epsilon:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    self.model_learner.eval()
                    action = self.model_learner(torch.tensor(o, device=self.device, dtype=torch.float).unsqueeze(0))\
                        .max(1)[1].detach().to('cpu').numpy()[0]
                o_, rew, done, info = env.step(action)
                ret += rew
                self.replay_buffer.insert(Transition(o, action, rew, o_, done))
                if self._wait():
                    self.elapsed_steps += 1
                    continue
                samples = self.replay_buffer.sample(self.mb_size)
                batch = Transition(*zip(*samples))
                non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), device=self.device)
                states = torch.tensor(batch.state, device=self.device, dtype=torch.float)
                actions = torch.tensor(batch.action, device=self.device, dtype=torch.long)
                rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
                non_final_states = torch.tensor([s for s, d in zip(batch.next_state, batch.done) if not d],
                                          device=self.device, dtype=torch.float)
                self.model_learner.train()
                outputs = self.model_learner(states).gather(1, actions.view(actions.size()[0], -1))
                if done:
                    targets = rewards
                else:
                    targets = torch.zeros(self.mb_size, device=self.device)
                    targets[non_final_mask] += self.gamma * self.model_target(non_final_states).max(1)[0].detach()
                    targets += rewards
                loss = self.criterion(outputs, targets.view(targets.size()[0], -1))
                self.optimizer.zero_grad()
                loss.backward()
                o = o_
                if self.grad_clamp:
                    for p in self.model_learner.parameters():
                        p.grad.data.clamp(*self.grad_clamp)
                self.optimizer.step()
                self.elapsed_steps += 1
                if self.elapsed_steps % self.target_update_frequency == 0:
                    self.model_target.load_state_dict(self.model_learner.state_dict())
                if self.epsilon_scheduler_use_steps:
                    self.epsilon_scheduler.step()
            returns.append(ret)
            self.elapsed_episodes += 1
            if not self.epsilon_scheduler_use_steps:
                self.epsilon_scheduler.step()
            if ep % 100 == 0 and len(returns) >= 100:
                print('mean prev 100 returns:', ep, ':', np.mean(returns[-100:]))
