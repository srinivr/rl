import torch.nn as nn
from models.base_model import BaseModel


class AtariModel(BaseModel):

    def __init__(self, n_channels=4, n_actions=None, embedding_dim=512):
        self.convolution_dim_out = 3136  # 7 * 7 * 64
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions

        super().__init__()
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.convolution_dim_out, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.n_actions)
        )

    def forward(self, x):
        x = self.convolution_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return self.output_tuple(x)

    @staticmethod
    def get_input_dimension():
        return 3
