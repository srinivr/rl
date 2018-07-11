import torch.nn as nn

from models.common.encoders.encoders import Encoder
from models.iterative.feature_models.base_iterative_model import BaseIterativeModel


class PushModel(BaseIterativeModel):

    def __init__(self, n_input_channels, n_actions, state_embedding=512, reward_embedding=64, reward_grounding=True,
                 model_grounding=False):

        super().__init__(n_input_channels, n_actions, state_embedding, reward_embedding, reward_grounding,
                         model_grounding)

        self.encoding_convolution = Encoder.get_push_encoder(self.n_input_channels)

        # features
        self.encoding_fc = nn.Sequential(
            nn.Linear(self.convolution_dim_out, self.state_embedding),
            nn.ReLU()
        )

    def _get_encoding(self, x):
        x = self.encoding_convolution(x)
        x = x.view(x.size(0), -1)
        return self.encoding_fc(x)

    @staticmethod
    def get_input_dimension():
        return 3

