from models.treeqn.treeqn_model import TreeQNModel
from collections import namedtuple
import torch.nn as nn


# TODO tree-reward ground and model grounding


class PushModel(TreeQNModel):

    def __init__(self, n_input_channels, n_actions, depth=2, state_embedding=128, reward_embedding=64, gamma=0.99,
                 lambd=0.8, reward_grounding=True, model_grounding=False):

        self.n_input_channels = n_input_channels
        self.convolution_dim_out = 48  # TODO automate this!
        super().__init__(n_actions, depth, state_embedding, reward_embedding, gamma, lambd, reward_grounding,
                         model_grounding)  # TODO understand what's under the hood of PyTorch of nn.Module
        self.encoding_convolution = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 24, kernel_size=3, stride=1),  # TODO make these as params?
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=1),
            nn.ReLU(),
        )

    def _get_encoding(self, x):
        x = self.encoding_convolution(x)
        x = x.view(x.size(0), -1)
        return self.encoding_fc(x)

    @staticmethod
    def get_input_dimension():
        return 3
