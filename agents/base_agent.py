import torch.nn as nn
import torch.optim as optim
from utils.scheduler.decay_scheduler import DecayScheduler
import numpy as np
import torch
from tensorboardX import SummaryWriter


class BaseAgent:

    def __init__(self, experiment_id, model_class, model_params, rng, device='cpu', training_evaluation_frequency=100,
                 optimizer=optim.RMSprop, optimizer_parameters={'lr': 1e-3, 'momentum': 0.9}, criterion=nn.SmoothL1Loss,
                 gamma=0.99, epsilon_scheduler=DecayScheduler(), epsilon_scheduler_use_steps=True,
                 target_synchronize_steps=1e4, parameter_update_steps=1, grad_clamp=None, auxiliary_losses=None):

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
        # self.parameter_update_steps = parameter_update_steps
        self.grad_clamp = grad_clamp
        self.td_losses = []
        self.auxiliary_losses = auxiliary_losses

        self.model_learner.to(self.device)
        self.model_target.to(self.device)
        self.optimizer = optimizer(self.model_learner.parameters(), **optimizer_parameters)
        self.model_target.load_state_dict(self.model_learner.state_dict())
        self.elapsed_model_steps = 0
        self.elapsed_env_steps = 0
        self.elapsed_episodes = 0
        self.writer = SummaryWriter(comment=experiment_id)

    def learn(self, env, eval_env=None, n_eval_episodes=100):
        raise NotImplementedError

    def _get_sample_action(self, env):
        raise NotImplementedError

    def _get_n_steps(self):
        raise NotImplementedError

    def _get_action_from_model(self, model, state, action_type):
        """
        :param: action_type: 'scalar' or 'list'
        :return: action from model, model outputs
        """
        assert action_type == 'list' or action_type == 'scalar'
        model.eval()
        model_in = torch.tensor(state, device=self.device, dtype=torch.float)
        for _ in range(self.model_class.get_input_dimension() + 1 - state.ndim):  # create a 2D tensor if input is 0D or 1D
            model_in = model_in.unsqueeze(0)
        model_out = model(model_in)
        actions = model_out.q_values.max(1)[1].detach().to('cpu').numpy()
        return actions if action_type == 'list' else actions[0]  # , model_out

    def _get_epsilon(self, *args):
        return self.epsilon_scheduler.get_epsilon(self.elapsed_model_steps) if self.epsilon_scheduler_use_steps \
            else self.epsilon_scheduler.get_epsilon(self.elapsed_episodes)

    def _get_epsilon_greedy_action(self, env, states):
        if np.random.random() < self._get_epsilon():
            action = self._get_sample_action(env)
        else:
            self.model_learner.eval()
            action = self._get_action_from_model(self.model_learner, states)
        o_, reward, done, info = env.step(action)
        self.elapsed_env_steps += self._get_n_steps()
        return action, o_, reward, done, info

    def _eval(self, env, n_episodes=100, action_type='scalar'):
        """
        :param action_type: 'scalar' or 'list' whichever is appropriate for the environment
        """
        returns = []
        self.model_target.eval()
        len = 0
        for ep in range(n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                len += 1
                action = self._get_action_from_model(self.model_target, o, action_type)
                o, rew, done, info = env.step(action)
                ret += rew
            returns.append(ret)
        print('mean eval return:', np.mean(returns), '..avg episode length:', len/n_episodes)
        print()
        self.writer.add_scalar('data/eval_loss', np.mean(returns), self.elapsed_env_steps)

    def _episode_updates(self):
        self.elapsed_episodes += 1
        if not self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _step_updates(self, states, actions, rewards, targets, batch_done):
        """
        Given a pytorch batch, update learner model, increment number of model updates
        (and possibly synchronize target model)

        :param batch_done: list
        """
        if states.size()[0] < 2:  # TODO do this only when batchnorm is used?
            print('Batch has only 1 example. Can cause problems if batch norm was used.. Skipping step')
            return
        self.model_learner.train()
        self.optimizer.zero_grad()
        model_outputs = self.model_learner(states)
        q_outputs = model_outputs.q_values.gather(1, actions.view(-1, 1))  # TODO remove
        loss2 = self.criterion(q_outputs, targets[0].view(-1, 1))  # TODO remove
        loss = torch.tensor(0.)
        for idx in range(len(self.td_losses)):
            loss = loss + self.td_losses[idx].get_loss(model_outputs, actions, targets[idx])
        if self.auxiliary_losses:
            for l in self.auxiliary_losses:
                loss += l.get_loss(model_outputs, actions, rewards, batch_done)
        loss.backward()
        if self.grad_clamp:
            for p in self.model_learner.parameters():
                p.grad.data.clamp(*self.grad_clamp)
        self.optimizer.step()
        self.elapsed_model_steps += 1
        if self.elapsed_model_steps % self.target_synchronize_steps == 0:
            self.model_target.load_state_dict(self.model_learner.state_dict())
            print('agents synchronized...at model step:', self.elapsed_model_steps, '.. env step:', self.elapsed_env_steps)
        if self.epsilon_scheduler_use_steps:
            self.epsilon_scheduler.step()

    def _get_batch(self, batch_states, batch_actions, batch_next_states, batch_rewards, batch_done, future_targets=None):
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
        #if (future_targets is None or None in future_targets) and not all(batch_done):
        if is_none and not all(batch_done):
            _next_non_final_states = torch.tensor(batch_next_states, **float_args)[non_final_mask]
            self.model_target.eval()
            model_outputs = self.model_target(_next_non_final_states)
        for idx in range(len(self.td_losses)):
            td_loss = self.td_losses[idx]
            # future_target = future_targets[idx]
            target = targets[idx]
            if future_targets is not None and future_targets[idx] is not None:
                target[non_final_mask] += self.gamma * future_targets[idx][non_final_mask]
            elif not all(batch_done):
                temp = target[non_final_mask]
                target[non_final_mask] += self.gamma * td_loss.get_bootstrap_values(model_outputs)
            target += td_loss.get_immediate_values(states, actions, rewards)
        return states, actions, rewards, targets, batch_done

    # TODO when doing linear decay
    #       epsilon decayed after every environment step (after buffer is adequately filled in DQN) but model step is
    #       used in scheduler.get_epsilon(). Is this consistent/clear?
