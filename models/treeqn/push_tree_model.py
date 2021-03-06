import torch.nn as nn
import numpy as np

from models.common.encoders.encoders import Encoder
from models.treeqn.treeqn_model import TreeQNModel
from utils.initializer import nn_init

# TODO model grounding


class PushTreeModel(TreeQNModel):

    def __init__(self, n_input_channels, n_actions, depth=2, state_embedding=128, reward_embedding=64, gamma=0.99,
                 lambd=0.8, reward_grounding=True, model_grounding=False):

        self.n_input_channels = n_input_channels
        self.convolution_dim_out = 48  # TODO automate this!
        super().__init__(n_actions, depth, state_embedding, reward_embedding, gamma, lambd, reward_grounding,
                         model_grounding)  # TODO what's under the hood of PyTorch of nn.Module
        self.encoding_convolution = Encoder.get_push_encoder(self.n_input_channels)
        self.encoding_fc = nn.Sequential(
            nn_init(nn.Linear(self.convolution_dim_out, self.state_embedding), w_scale=np.sqrt(2)),
            nn.ReLU()
        )

    def _get_encoding(self, x):
        x = self.encoding_convolution(x)
        x = x.view(x.size(0), -1)  # batch size x (c x h x w)
        return self.encoding_fc(x)

    @staticmethod
    def get_input_dimension():
        return 3
