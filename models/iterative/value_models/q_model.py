import torch.nn as nn

from models.base_model import BaseModel


class QModel(BaseModel):

    def __init__(self, input_embedding_dim, state_embedding_dim, n_actions, normalize_embeddings=False):
        self.input_embedding_dim = input_embedding_dim
        self.state_embedding_dim = state_embedding_dim
        self.n_actions = n_actions
        self.normalize_embeddings = normalize_embeddings
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(nn.Sequential(
                nn.Linear(self.input_embedding_dim, self.state_embedding_dim),
                nn.ReLU()))
        self.q_fn = nn.Linear(self.state_embedding_dim, self.n_actions)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            if self.normalize_embeddings:
                x = x/x.norm(dim=1, keepdim=True)
        x = self.q_fn(x)
        return self.output_tuple(x)
        # return self.output_tuple(self.layers(x))

    @staticmethod
    def get_input_dimension():
        return 1
