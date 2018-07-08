from collections import namedtuple
from utils.initializer import nn_init, xav_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel


class TreeQNModel(BaseModel):
    def __init__(self, n_actions, depth=2, state_embedding=128, reward_embedding=64, gamma=0.99,
                 lambd=0.8, reward_grounding=False, model_grounding=False):
        self.n_actions = n_actions
        self.depth = depth
        self.gamma = gamma
        self.lambd = lambd
        self.state_embedding = state_embedding
        self.reward_embedding = reward_embedding
        self.reward_grounding = reward_grounding
        self.model_grounding = model_grounding
        self.reward_tuple = namedtuple('Rewards', 'rewards next_rewards')
        tuple_attributes = []
        if self.reward_grounding:
            tuple_attributes.append('rewards')
        if self.model_grounding:
            tuple_attributes.append('model_prediction')
        super().__init__(tuple_attributes)

        self.encoding_fc = nn.Sequential(
            nn_init(nn.Linear(self.convolution_dim_out, self.state_embedding), w_scale=np.sqrt(2)),
            nn.ReLU()
        )
        self.action_independent_transition = nn.Sequential(
            nn.Linear(self.state_embedding, self.state_embedding),
            nn.Tanh()
        )
        # create actions * some units
        self.action_transition = nn.ModuleList()
        for _ in range(self.n_actions):
            self.action_transition.append(nn.Sequential(
                xav_init(nn.Linear(self.state_embedding, self.state_embedding, bias=False)),
                # turn off bias for action dependent transition
                nn.Tanh()
            ))
        self.reward_fn = nn.Sequential(
            nn_init(nn.Linear(self.state_embedding, 64), w_scale=np.sqrt(2)),
            nn.ReLU(),
            nn_init(nn.Linear(64, self.n_actions), w_scale=0.01)
        )
        self.value_fn = nn.Sequential(
            nn_init(nn.Linear(self.state_embedding, 1), w_scale=0.01)
        )

    def forward(self, x):
        x = self._get_encoding(x)
        x = x / x.norm(dim=1, keepdim=True)  # normalize embedding
        q_values, rewards = self._q_a(x, 0)
        if self.reward_grounding:
            return self.output_tuple(q_values, rewards)
        return self.output_tuple(q_values)

    def _q_a(self, s, depth):
        """
        perform action independent + dependent transition and compute the backup
        :param s: normalized state embedding
        :return : Q(s, a) dim: b x a
        """
        reward = self.reward_fn(s)
        s = s + self.action_independent_transition(s)
        v_n = []
        transition_rewards = []
        for idx in range(self.n_actions):
            transition = s + self.action_transition[idx](s)  # residual connection
            transition = transition / transition.norm(dim=1, keepdim=True)  # normalize the transition state
            v, transition_return = self._v_fn(transition, depth + 1)
            v_n.append(v)
            transition_rewards.append(transition_return)
        v_n = torch.cat(v_n, dim=1)
        return reward + self.gamma * v_n, self.reward_tuple(reward, transition_rewards)

    def _v_fn(self, s, depth):
        """
        :param s: normalized state embedding
        :return: 1 x 1
        """
        if depth == self.depth:
            return self.value_fn(s), None
        action_values, transition_rewards = self._q_a(s, depth)
        backup = self._backup_fn(action_values)  # assert shape is b x 1
        return (1. - self.lambd) * self.value_fn(s) + self.lambd * backup, transition_rewards

    def _backup_fn(self, q_a):
        return torch.sum(F.softmax(q_a, dim=1) * q_a, dim=1, keepdim=True)

    def _get_encoding(self, x):
        raise NotImplementedError

    @staticmethod
    def get_input_dimension():
        raise NotImplementedError

