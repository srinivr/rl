from models.base_model import BaseModel
from collections import namedtuple
import torch.nn as nn
import numpy as np

from utils.initializer import nn_init


class SimpleCartPoleModel(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = 24
        self.fc_layers = nn.Sequential(
            nn_init(nn.Linear(self.input_dim, self.embedding), w_scale=np.sqrt(2)),
            # nn.BatchNorm1d(self.embedding),
            # nn.ReLU(),
            nn.Tanh(),
            nn_init(nn.Linear(self.embedding, self.embedding), w_scale=np.sqrt(2)),
            # nn.BatchNorm1d(self.embedding),
            # nn.ReLU(),
            nn.Tanh(),
            nn_init(nn.Linear(self.embedding, self.output_dim))
        )

    def forward(self, x):
        return self.output_tuple(self.fc_layers(x))

    @staticmethod
    def get_input_dimension():
        return 1

