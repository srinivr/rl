import os
import shutil

import torch.nn as nn
import torch.optim as optim
from utils.scheduler.decay_scheduler import DecayScheduler
import numpy as np
import torch

from torch.optim.lr_scheduler import LambdaLR
from collections import namedtuple
from tensorboardX import SummaryWriter


# TODO learn episodes/steps inconsistent in dqn and n-step dqn

class BaseAgent:

    def __init__(self, model_class, model_params, rng, device='cpu', training_evaluation_frequency=100,
                 optimizer=optim.RMSprop, optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, lr_scheduler_fn=None,
                 criterion=nn.SmoothL1Loss, gamma=0.99, epsilon_scheduler=DecayScheduler(),
                 epsilon_scheduler_use_steps=True, target_synchronize_steps=1e4, td_losses=None, grad_clamp=None,
                 grad_clamp_parameters=None, auxiliary_losses=None, input_transforms=None, output_transforms=None,
                 auxiliary_env_info=None, log=True, log_dir=None, save_checkpoint=True):

        self.model_class = model_class
        self.rng = rng
        self.device = device
        self.training_evaluation_frequency = training_evaluation_frequency  # steps or episodes depending on learn()
        self.criterion = criterion()
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_scheduler_use_steps = epsilon_scheduler_use_steps
        self.model_learner = self.model_class(*model_params)
        self.model_target = self.model_class(*model_params)
        self.target_synchronize_steps = target_synchronize_steps  # global steps across processes
        self.td_losses = [] if td_losses is None else td_losses
        self.grad_clamp = grad_clamp
        if self.grad_clamp:
            assert grad_clamp_parameters is not None and (grad_clamp_parameters == 'value' or 'norm')
            self.grad_clamp_parameters = grad_clamp_parameters
        self.auxiliary_losses = [] if auxiliary_losses is None else auxiliary_losses
        self.input_transforms = [] if input_transforms is None else input_transforms
        self.output_transforms = [] if output_transforms is None else output_transforms

        self.model_learner.to(self.device)
        self.model_target.to(self.device)
        self.optimizer = optimizer(self.model_learner.parameters(), **optimizer_parameters)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_scheduler_fn) if lr_scheduler_fn else None
        self.model_target.load_state_dict(self.model_learner.state_dict())
        self.elapsed_model_steps = 0
        self.elapsed_env_steps = 0
        self.elapsed_episodes = 0
        self.auxiliary_env_info = auxiliary_env_info
        if self.auxiliary_env_info:  # namedtuple containing names and types each holding a list
            self.auxiliary_tuple = namedtuple('auxiliary', self.auxiliary_env_info.names)
        self.log = log
        self.save_checkpoint = save_checkpoint
        if self.log or self.save_checkpoint:
            assert log_dir is not None
        if self.log:
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir=log_dir)
        self._best_return = None
        self._is_best = None

    def learn(self, env, eval_env=None, n_learn_iterations=None, n_eval_episodes=100):
        raise NotImplementedError

    def _get_sample_action(self, env):
        raise NotImplementedError

    def _get_n_steps(self):
        raise NotImplementedError

    def add_input_transform(self, input_transform):
        self.input_transforms.append(input_transform)

    def add_output_transform(self, output_transform):
        self.output_transforms.append(output_transform)

    def get_learner_model(self):
        return self.model_learner

    def get_target_model(self):  # return immutable? if returning how to immutable ensure that callers have latest copy?
        return self.model_target

    def evaluate(self, model, state):
        """
        forward pass in evaluation mode
        """
        model.eval()
        inputs = torch.tensor(state, device=self.device, dtype=torch.float)
        if isinstance(state, torch.Tensor):
            dim = state.dim()
        else:
            dim = state.ndim
        for _ in range(self.model_class.get_input_dimension() + 1 - dim):  # create a nd tensor if input < nd
            inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            model_outputs = model(inputs)
        return model_outputs

    def _apply_input_transform(self, states):
        for input_transform in self.input_transforms:
            states = input_transform.transform(states)
        return states

    def _apply_output_transform(self, states, model_outputs):
        for output_transform in self.output_transforms:
            model_outputs = output_transform.transform(states, model_outputs)
        return model_outputs

    def _get_epsilon(self):
        return self.epsilon_scheduler.get_epsilon()

    def _get_greedy_action(self, model, state, action_type):  # model.eval() within the function
        """
        :param: action_type: 'scalar' or 'list'
        :return: action from model
        """
        assert action_type == 'list' or action_type == 'scalar'
        model_output = self.evaluate(model, state)
        model_output = self._apply_output_transform(state, model_output)
        actions = model_output.q_values.max(1)[1].detach().to('cpu').numpy()
        return actions if action_type == 'list' else actions[0]

    def _get_epsilon_greedy_action(self, env, states, *args):
        if np.random.random() < self._get_epsilon():
            action = self._get_sample_action(env)
        else:
            self.model_learner.eval()
            action = self._get_greedy_action(self.model_learner, states)
        return action

    def _get_epsilon_greedy_action_and_step(self, env, states, *args):  # states must be input-transformed
        action = self._get_epsilon_greedy_action(env, states, args)
        o_, reward, done, info, *auxiliary_info = env.step(action)
        self.elapsed_env_steps += self._get_n_steps()
        o_ = self._apply_input_transform(o_)
        return (action, o_, reward, done, info, *auxiliary_info)

    def _eval(self, env, n_episodes=100, action_type='scalar'):
        """
        :param action_type: 'scalar' or 'list' whichever is appropriate for the environment
        """
        returns = []
        self.model_target.eval()
        length = 0.
        for ep in range(n_episodes):
            o = env.reset()
            done = False
            ret = 0.
            while not done:
                length += 1
                o = self._apply_input_transform(o)
                action = self._get_greedy_action(self.model_target, o, action_type)
                o, rew, done, info, *auxiliary_info = env.step(action)
                ret += rew
            returns.append(ret)
        print('mean eval return:', np.mean(returns), '..avg episode length:', length / n_episodes)
        print()
        if self.log:
            self.writer.add_scalar('data/eval_rewards', np.mean(returns), self.elapsed_env_steps)
            self.writer.add_scalar('data/eval_episode_length', length / n_episodes, self.elapsed_env_steps)
        if self._best_return is None or np.mean(returns) > self._best_return:
            self._best_return = np.mean(returns)
            self._is_best = True
        else:
            self._is_best = False
        if self.save_checkpoint:
            self._save()

    def _episode_updates(self):
        self.elapsed_episodes += 1
        if not self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _step_updates(self, states, actions, rewards, targets, batch_done, auxiliary_info):
        """
        Given a pytorch batch, update learner model, increment number of model updates
        (and possibly synchronize target model)

        :param states: transformed input states
        :param batch_done: list
        :param auxiliary_info: a namedtuple containing pytorch tensors
        """
        if states.size()[0] < 2:  # TODO do this only when batchnorm is used?
            print('Batch has only 1 example. Can cause problems if batch norm was used.. Skipping step')
            return
        self.model_learner.train()
        model_outputs = self.model_learner(states)
        assert len(self.td_losses) + len(self.auxiliary_losses) != 0  # to train model at least one loss is required
        loss = torch.tensor(0., device=self.device)
        for idx in range(len(self.td_losses)):
            _loss = self.td_losses[idx].get_loss(model_outputs, actions, targets[idx])
            loss += _loss
            if self.log:
                self.writer.add_scalar('data/td_loss/' + self.td_losses[idx].get_name(), _loss, self.elapsed_env_steps)
        if self.auxiliary_losses:
            for l in self.auxiliary_losses:
                _loss = l.get_loss(model_outputs, actions, rewards, batch_done, auxiliary_info)
                loss += _loss
                if self.log:
                    self.writer.add_scalar('data/auxiliary_loss/' + l.get_name(), _loss, self.elapsed_env_steps)
        if self.log:
            self.writer.add_scalar('data/cumulative_loss', loss, self.elapsed_env_steps)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clamp:
            if self.grad_clamp == 'value':
                for p in self.model_learner.parameters():
                    p.grad.data.clamp(*self.grad_clamp_parameters)
            elif self.grad_clamp == 'norm':
                grad_norm = nn.utils.clip_grad_norm_(self.model_learner.parameters(), *self.grad_clamp_parameters)
                if self.log:
                    self.writer.add_scalar('data/grad_norm', grad_norm, self.elapsed_env_steps)
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.optimizer.step()
        self.elapsed_model_steps += 1
        if self.elapsed_model_steps % self.target_synchronize_steps == 0:
            self.model_target.load_state_dict(self.model_learner.state_dict())
            print('agents synchronized... model step:', self.elapsed_model_steps, '. env step:', self.elapsed_env_steps)
        if self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()
        if self.log:
            self._log_values()

    def _get_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done,
                   future_targets=None):
        """
        Construct pytorch batch tensors for _step_updates.
        :param batch_actions: 1D
        :param future_target: if not None, use it as next step backup value (useful for n-step updates)
        """
        float_args = dict(device=self.device, dtype=torch.float)
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch_done)), device=self.device)
        states = torch.tensor(batch_states, **float_args)
        actions = torch.tensor(batch_actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(batch_rewards, **float_args)
        targets = [torch.zeros(len(actions), *td_loss.get_shape(), device=self.device) for td_loss in self.td_losses]
        # TODO [HIGH] both n-step and 1-step target for different losses;
        # None in list is not supported (yet) in a list containing pytorch tensors
        is_none = future_targets is None
        if not is_none:
            for ft in future_targets:
                is_none = is_none or ft is None
                if is_none:
                    break
        if is_none and not all(batch_done):
            _next_non_final_states = torch.tensor(batch_next_states, **float_args)[non_final_mask]
            with torch.no_grad():
                self.model_target.eval()
            model_outputs = self.model_target(_next_non_final_states)
        for idx in range(len(self.td_losses)):
            td_loss = self.td_losses[idx]
            # future_target = future_targets[idx]
            target = targets[idx]
            if future_targets is not None and future_targets[idx] is not None:
                target[non_final_mask] += self.gamma * future_targets[idx][non_final_mask]
            elif not all(batch_done):
                target[non_final_mask] += self.gamma * td_loss.get_bootstrap_values(model_outputs)
            target += td_loss.get_immediate_values(states, actions, rewards)
        return states, actions, rewards, targets, batch_done

    def _get_auxiliary_batch(self, *auxiliary_info):
        """

        :param auxiliary_info: list containing iterable
        :return: [] if auxiliary is empty else namedtuple
        """
        if len(auxiliary_info) == 0:
            return []
        result = self.auxiliary_tuple(*zip(*auxiliary_info))
        temp_result = []
        for name, dtype in zip(self.auxiliary_env_info.names, self.auxiliary_env_info.types):
            temp_result.append(torch.tensor(getattr(result, name), device=self.device, dtype=dtype))
        return self.auxiliary_tuple(*temp_result)

    # TODO when doing linear decay
    #       epsilon decayed after every environment step (after buffer is adequately filled in DQN) but model step is
    #       used in scheduler.get_epsilon(). Is this consistent/clear?

    def _log_values(self):
        if self.log:
            self.writer.add_scalar('data/epsilon', self.epsilon_scheduler.get_epsilon(), self.elapsed_env_steps)
            # self.writer.add_scalar('data/lr', self.optimizer.l)  # TODO

    def _get_state(self):
        return {
            'elapsed_env_steps': self.elapsed_env_steps,
            'elapsed_model_steps': self.elapsed_model_steps,
            'state_dict': self.model_target.state_dict(),
            'best_return': self._best_return,
            'optimizer': self.optimizer.state_dict()
        }

    def _set_state(self, state):
        self.elapsed_env_steps = state['elapsed_env_steps']
        self.elapsed_model_steps = state['elapsed_model_steps']
        self.model_learner.load_state_dict(state['state_dict'])
        self.model_target.load_state_dict(self.model_learner.state_dict())
        # self.model_learner.to(self.device)  # TODO is this necessary?
        # self.model_target.to(self.device)
        self._best_return = state['best_return']
        self.optimizer.load_state_dict(state['optimizer'])
        self.epsilon_scheduler.set_elapsed_steps(self.elapsed_env_steps)

    def _save(self):
        state = self._get_state()
        filename = os.path.join(self.log_dir, 'checkpoint_' + str(self.elapsed_env_steps) + '.ckpt')
        torch.save(state, filename)
        if self._is_best:
            shutil.copyfile(filename, os.path.join(self.log_dir, 'best.ckpt'))

    def load(self, path, idx, load_best=False):
        path = os.path.join(path, 'best.ckpt') if load_best else os.path.join(path, 'checkpoint_' + str(idx) + '.ckpt')
        if os.path.isfile(path):
            state = torch.load(path)
            self._set_state(state)
            print("=> loaded checkpoint '{}' (elapsed env steps {})".format(path, state['elapsed_env_steps']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            raise ValueError
