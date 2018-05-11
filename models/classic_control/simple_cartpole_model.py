from models.base_model import BaseModel
from collections import namedtuple
import torch.nn as nn


class SimpleCartPoleModel(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = 24
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding),
            nn.BatchNorm1d(self.embedding),
            nn.ReLU(),
            nn.Linear(self.embedding, self.embedding),
            nn.BatchNorm1d(self.embedding),
            nn.ReLU(),
            nn.Linear(self.embedding, self.output_dim)
        )

    def forward(self, x):
        return self.output_tuple(self.fc_layers(x))

    @staticmethod
    def get_input_dimension():
        return 1

