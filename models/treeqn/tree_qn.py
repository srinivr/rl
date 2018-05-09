import torch
import torch.nn as nn


class PushModel(nn.Module):
    def __init__(self, n_input_channels, n_actions, depth=2, state_embedding=128, reward_embedding=64, gamma=0.99,
                 lambd=1):
        super(PushModel, self).__init__()
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.depth = depth
        self.gamma = gamma
        self.lambd = lambd
        self.state_embedding = state_embedding
        self.reward_embedding = reward_embedding
        self.convolution_dim_out = 48  # TODO automate this!
        self.softmax = nn.Softmax(dim=1)
        self.encoding = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 24, kernel_size=3, stride=1),  # TODO make these as params
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.convolution_dim_out, 128),
            nn.ReLU())
        self.action_independent_transition = nn.Sequential(
            nn.Linear(self.state_embedding, self.state_embedding),
            nn.ReLU()
        )
        # create actions * some units
        self.action_transition = []
        for _ in range(self.n_actions):
            self.action_transition.append(nn.Sequential(
                nn.Linear(self.state_embedding, self.state_embedding),
                nn.ReLU()
            ))
        self.reward_fn = nn.Sequential(
            nn.Linear(self.state_embedding, self.reward_embedding),
            nn.ReLU(),
            nn.Linear(self.reward_embedding, self.n_actions)
        )
        self.value_fn = nn.Sequential(
            nn.Linear(self.state_embedding, 1)
        )

    def forward(self, x):
        # TODO residual connections
        x = self.encoding(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = self.action_independent_transition(x)
        return self._q_a(x, 0)

    def _backup_fn(self, q_a):
        return torch.sum(self.softmax(q_a) * q_a, dim=1).view(-1, 1)  # create 2D

    def _q_a(self, s, depth):
        """
        perform action dependent transition and compute the backup
        """
        r_n = self.reward_fn(s)
        v_n = []
        for idx in range(self.n_actions):
            transition = self.action_transition[idx](s)
            transition = transition / transition.norm()
            v = self._v_fn(transition, depth + 1)
            v_n.append(v)
        v_n = torch.cat(v_n, dim=1)
        return r_n + self.gamma * v_n

    def _v_fn(self, s, depth):
        if depth == self.depth:
            return self.value_fn(s)
        bkp = self._backup_fn(self._q_a(s, depth))  # assert b x 1
        return (1. - self.lambd) * self.value_fn(s) + self.lambd * bkp
