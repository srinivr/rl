import torch.nn as nn
import torch.optim as optim
from utils.scheduler.decay_scheduler import StepDecayScheduler
import numpy as np
import torch


class BaseAgent:

    def __init__(self, model_class, model_params, rng, device='cpu', n_episodes=2000, evaluation_frequency=100, lr=1e-3,
                 momentum=0.9, criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99,
                 epsilon_scheduler=StepDecayScheduler(), epsilon_scheduler_use_steps=True, target_update_steps=1e4,
                 parameter_update_frequency=1, grad_clamp=None):

        self.model_class = model_class
        self.rng = rng
        self.device = device
        self.n_episodes = n_episodes
        self.evaluation_frequency = evaluation_frequency  # steps or episodes depending on the context! TODO improve?
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
        self.elapsed_model_steps = 0  # number of updates to the model! # TODO this may not be right! every n-steps in n-step
        self.elapsed_env_steps = 0  # number of interactions with the/an environment
        self.elapsed_episodes = 0

    def learn(self, env, eval_env=None):
        raise NotImplementedError

    def _get_sample_action(self, env):
        raise NotImplementedError

    def _get_action_from_model(self, model, o, action_type):
        """
        :param action_type: 'scalar' or 'list'
        """
        assert action_type == 'list' or action_type == 'scalar'
        model.eval()
        model_in = torch.tensor(o, device=self.device, dtype=torch.float)
        for _ in range(2 - o.ndim):  # create a 2D tensor if input is 0D or 1D
            model_in = model_in.unsqueeze(0)
        model_out = model(model_in).max(1)[1].detach().to('cpu').numpy()
        return model_out if action_type == 'list' else model_out[0]

    def _get_epsilon(self, *args):
        return self.epsilon_scheduler.get_epsilon(self.elapsed_model_steps) if self.epsilon_scheduler_use_steps \
            else self.epsilon_scheduler.get_epsilon(self.elapsed_episodes)

    def _get_epsilon_greedy_action(self, env, o):
        if np.random.random() < self._get_epsilon():
            action = self._get_sample_action(env)
        else:
            self.model_learner.eval()
            action = self._get_action_from_model(self.model_learner, o)
        o_, reward, done, info = env.step(action)
        self.elapsed_env_steps += 1
        return action, o_, reward, done, info

    def _eval(self, env, n_episodes=100, action_type='scalar'):
        """
        :param action_type: 'scalar' or 'list' whichever is appropriate for the environment
        """
        returns = []
        self.model_target.eval()
        for ep in range(n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                action = self._get_action_from_model(self.model_target, o, action_type)
                o, rew, done, info = env.step(action)
                ret += rew
            returns.append(ret)
        print('mean eval return:', np.mean(returns))
        print()

    def _episode_updates(self):
        self.elapsed_episodes += 1
        if not self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _step_updates(self, states, actions, targets):  # inputs: pytorch tensors
        """
        Given a batch, update learner model, increment number of model updates (and possibly synchronize target model)
        """
        if states.size()[0] < 2:  # TODO do this only when batchnorm is used?
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
            print('agents synchronized...at model step:', self.elapsed_model_steps, '.. env step:', self.elapsed_env_steps)
        if self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _get_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, future_target=None):
        """
        Construct pytorch batch tensors for _step_updates.
        :param future_target: if not None, use it as next step backup value (useful for n-step updates)
        """
        float_args = dict(device=self.device, dtype=torch.float)
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch_done)), device=self.device)
        states = torch.tensor(batch_states, **float_args)
        actions = torch.tensor(batch_actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(batch_rewards, **float_args)
        targets = torch.zeros(len(actions), device=self.device)
        if future_target is not None:
            targets[non_final_mask] += self.gamma * future_target[non_final_mask]
        elif not all(batch_done):
            _next_non_final_states = torch.tensor(batch_next_states, **float_args)[non_final_mask]
            # TODO check below!
            # next_non_final_states = torch.tensor([s for s, d in zip(batch_next_states, batch_done) if not d], **float_args)
            # print('_nnfs', len(_next_non_final_states), 'nnfs:', len(next_non_final_states), torch.sum(_next_non_final_states - next_non_final_states))
            self.model_target.eval()
            targets[non_final_mask] += self.gamma * self.model_target(_next_non_final_states).max(1)[0].detach()
        targets += rewards
        return states, actions, targets

    # TODO when doing linear decay
    #       epsilon decayed after every environment step (after buffer is adequately filled in DQN) but model step is
    #       used in scheduler.get_epsilon(). Is this consistent/clear?
