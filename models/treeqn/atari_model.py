import torch.nn as nn
import numpy as np

from models.treeqn.treeqn_model import TreeQNModel
from utils.initializer import nn_init


class AtariTreeModel(TreeQNModel):
    def __init__(self, n_input_channels=4, n_actions=None, depth=2, state_embedding=512, reward_embedding=64, gamma=0.99,
                 lambd=0.8, reward_grounding=True, model_grounding=False):
        self.convolution_dim_out = 2592  # 9 * 9 * 32
        super().__init__(n_actions, depth, state_embedding, reward_embedding, gamma, lambd, reward_grounding,
                         model_grounding)

        self.encoding_convolution = nn.Sequential(
            nn_init(nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=8, stride=4), w_scale=np.sqrt(2)),
            nn.ReLU(inplace=True),
            nn_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2), w_scale=np.sqrt(2)),
            nn.ReLU(inplace=True)
        )

        self.encoding_fc = nn.Sequential(
            nn_init(nn.Linear(self.convolution_dim_out, self.state_embedding), w_scale=np.sqrt(2)),
            nn.ReLU(inplace=True)
        )

    def _get_encoding(self, x):
        x = self.encoding_convolution(x)
        x = x.view(x.size(0), -1)  # batch size x (c x h x w)
        return self.encoding_fc(x)

    @staticmethod
    def get_input_dimension():
        return 3
