import itertools
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from agents.base_agent import BaseAgent
from utils.scheduler.decay_scheduler import DecayScheduler
import numpy as np


class WrongNStepSynchronousDQNAgent(BaseAgent):
    """
    https://arxiv.org/pdf/1710.11417.pdf (batched n-step)
    """

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000, lr=1e-3, momentum=0.9,
                 criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99, epsilon_scheduler=DecayScheduler(),
                 epsilon_scheduler_use_steps=True, target_synchronize_steps=1e4, parameter_update_frequency=1,
                 grad_clamp=None, n_step=5, n_envs=1):

        self.n_step = n_step
        self.n_envs = n_envs
        self.elapsed_env_step = np.zeros(self.n_envs)
        self.batch_values = namedtuple('Values', 'done step_ctr rewards states actions targets')
        super().__init__(model_class, model_params, rng, device, n_episodes, lr, momentum, criterion, optimizer, gamma,
                         epsilon_scheduler, epsilon_scheduler_use_steps, target_synchronize_steps,
                         parameter_update_frequency, grad_clamp)

    def learn(self, envs):
        float_args = dict(device=self.device, dtype=torch.float)
        for ep in range(self.n_episodes):
            batch_observations = [envs[i].reset() for i in range(self.n_envs)]
            batch_done = [False] * self.n_envs
            batch_returns = 0.  # np.empty(shape=(self.n_envs, self.n_episodes))
            print('batch observations:', batch_observations)
            while not all(batch_done):
                """
                    multiprocessing discussion:
                     https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
                """
                done_event = mp.Event()
                queue = mp.Queue()
                processes = []
                for i in range(self.n_envs):
                    p = mp.Process(target=self._step, args=(done_event, queue, i, i, envs[i], batch_observations[i],
                                                            batch_done[i],))
                    p.start()
                    processes.append(p)
                finished = 0
                results = []
                while finished != self.n_envs:
                    result = queue.get()
                    if result is None:
                        finished += 1
                    else:
                        results.append(result)
                results = self.batch_values(*zip(*results))  # TODO the state of env will not be changed.
                batch_returns += np.sum(results.rewards)
                batch_done = list(results.done)
                print('batch_done', batch_done, 'all done?', all(batch_done), 'steps', results.step_ctr)
                print('cumulative steps', self.elapsed_env_step)
                step_counters = results.step_ctr
                _states = list(itertools.chain.from_iterable(results.states))  # list of numpy arrays
                _actions = list(itertools.chain.from_iterable(results.actions))  # list of numpy arrays
                targets = torch.cat(list(results.targets))  # list of variables on the device
                self.elapsed_model_steps += np.sum(step_counters)  # TODO _step_updates increment the counter by 1
                states = torch.tensor(_states, **float_args)
                actions = torch.tensor(_actions, device=self.device, dtype=torch.long)
                self._step_updates(states, actions, targets)
                batch_observations = _states
            # print returns
            print('average return:', np.mean(batch_returns))
            self._episode_updates()

    def _step(self, done_event, q, env_idx, seed, env, o, done):
        """
        1 n-step step (not the usual n-step)
        :param env_idx:
        :param seed:
        :param env:
        :param done:
        :return:
        """
        print('step called')
        np.random.seed(seed)
        step_ctr = 0
        rewards = []
        states = []
        actions = []
        while not done and step_ctr < self.n_step:
            action, o_, reward, done, info = self._get_epsilon_greedy_action(env, o, env_idx)
            step_ctr += 1
            self.elapsed_env_step[env_idx] += 1  # won't have any effect TODO remove
            states.append(o)
            rewards.append(reward)
            actions.append(action)
            o = o_
        targets = torch.empty(step_ctr, 1)
        if done:
            target = reward
        else:
            self.model_target.eval()
            target = reward + self.gamma * self.model_target(torch.tensor(o_, device=self.device, dtype= \
                torch.float).unsqueeze(0)).max(1)[0].detach()
        targets[step_ctr - 1] = target
        for n in range(step_ctr - 2, -1, -1):
            targets[n] = rewards[n] + self.gamma * targets[n + 1]
        q.put((done, step_ctr, np.sum(rewards), states, actions, targets))
        q.put(None)
        done_event.wait()

    def _get_epsilon(self, arg):
        epsilon = self.epsilon_scheduler.get_epsilon(self.elapsed_env_step[arg])
        return epsilon
