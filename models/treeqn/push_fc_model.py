import torch.nn as nn

from models.base_model import BaseModel
from models.common.encoders.encoders import Encoder


class PushFCModel(BaseModel):

    def __init__(self, n_input_channels, n_actions, state_embedding=512):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.state_embedding = state_embedding
        self.convolution_dim_out = 48
        super().__init__()
        self.encoding = Encoder.get_push_encoder(self.n_input_channels)
        self.layers = nn.Sequential(
            nn.Linear(self.convolution_dim_out, self.state_embedding),
            nn.ReLU(),
            nn.Linear(self.state_embedding, self.state_embedding),
            nn.ReLU(),
            nn.Linear(self.state_embedding, self.state_embedding),
            nn.ReLU(),
            nn.Linear(self.state_embedding, self.n_actions)
        )

    def forward(self, x):
        x = self.encoding(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.output_tuple(x)

    @staticmethod
    def get_input_dimension():
        return 3
