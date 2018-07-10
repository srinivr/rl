import torch.nn as nn

from models.base_model import BaseModel


class QModel(BaseModel):

    def __init__(self, input_embedding_dim, state_embedding_dim, n_actions):
        self.input_embedding_dim = input_embedding_dim
        self.state_embedding_dim = state_embedding_dim
        self.n_actions = n_actions
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.input_embedding_dim, self.state_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.state_embedding_dim, self.state_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.state_embedding_dim, self.state_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.state_embedding_dim, self.n_actions)
        )

    def forward(self, x):
        return self.output_tuple(self.layers(x))

    @staticmethod
    def get_input_dimension():
        return 1
