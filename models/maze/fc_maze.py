import torch.nn as nn

from models.base_model import BaseModel


class FCMaze(BaseModel):
    def __init__(self, input_dim, output_dim, encoding_dim=128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoding_dim = encoding_dim
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.encoding_dim),
            nn.ReLU(),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.ReLU(),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, self.output_dim)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return self.output_tuple(self.fc_layer(x))

    @staticmethod
    def get_input_dimension():
        return 2
