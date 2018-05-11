from collections import namedtuple

from models.base_model import BaseModel
import torch
import torch.nn as nn

# TODO model grounding


class PushModel(BaseModel):

    def __init__(self, n_input_channels, n_actions, depth=2, state_embedding=128, reward_embedding=64, gamma=0.99,
                 lambd=0.8, reward_grounding=True, model_grounding=False):
        self.reward_grounding = reward_grounding
        self.model_grounding = model_grounding
        self.reward_tuple = namedtuple('Rewards', 'rewards next_rewards')
        tuple_attributes = []
        if self.reward_grounding:
            tuple_attributes.append('rewards')
        if self.model_grounding:
            tuple_attributes.append('model_prediction')
        super().__init__(tuple_attributes)
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.depth = depth
        self.gamma = gamma
        self.lambd = lambd
        self.state_embedding = state_embedding
        self.reward_embedding = reward_embedding
        self.convolution_dim_out = 48  # TODO automate this!
        self.softmax = nn.Softmax(dim=1)
        self.encoding_conv = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 24, kernel_size=3, stride=1),  # TODO make these as params
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.encoding_fc = nn.Sequential(
            nn.Linear(self.convolution_dim_out, 128),
            # nn.ReLU()
        )
        self.action_independent_transition = nn.Sequential(
            nn.Linear(self.state_embedding, self.state_embedding),
            nn.Tanh()
        )
        # create actions * some units
        self.action_transition = nn.ModuleList()
        for _ in range(self.n_actions):
            self.action_transition.append(nn.Sequential(
                nn.Linear(self.state_embedding, self.state_embedding, bias=False),  # turn off bias for action dependent transition
                nn.Tanh()
            ))
        self.reward_fn = nn.Sequential(
            nn.Linear(self.state_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        self.value_fn = nn.Sequential(
            nn.Linear(self.state_embedding, 1)
        )

    def forward(self, x):
        x = self.encoding_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoding_fc(x)
        x = x + self.action_independent_transition(x)  # residual connection
        q_values, rewards = self._q_a(x, 0)
        if self.reward_grounding:
            return self.output_tuple(q_values, rewards)
        return self.output_tuple(q_values)

    def _backup_fn(self, q_a):
        return torch.sum(self.softmax(q_a) * q_a, dim=1).view(-1, 1)  # create 2D

    def _q_a(self, s, depth):  # return dim: b x a
        """
        perform action dependent transition and compute the backup
        """
        reward = self.reward_fn(s)
        v_n = []
        transition_rewards = []
        for idx in range(self.n_actions):
            transition = s + self.action_transition[idx](s)  # residual connection
            transition = transition / transition.norm()  # normalize the transition state
            v, transition_return = self._v_fn(transition, depth + 1)
            v_n.append(v)
            transition_rewards.append(transition_return)
        v_n = torch.cat(v_n, dim=1)
        return reward + self.gamma * v_n, self.reward_tuple(reward, transition_rewards)

    def _v_fn(self, s, depth):  # return dim: 1 x 1
        if depth == self.depth:
            return self.value_fn(s), None
        action_values, transition_rewards = self._q_a(s, depth)
        backup = self._backup_fn(action_values)  # assert shape is b x 1
        return (1. - self.lambd) * self.value_fn(s) + self.lambd * backup, transition_rewards

    def get_output_namedtuple(self):
        return self.output_tuple

    @staticmethod
    def get_input_dimension():
        return 3