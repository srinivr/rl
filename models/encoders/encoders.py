import torch.nn as nn

from utils.initializer import nn_init


class Encoder:

    @staticmethod
    def get_push_encoder(n_input_channels):
        encoder = nn.Sequential(
            nn_init(nn.Conv2d(n_input_channels, 24, kernel_size=3, stride=1), w_scale=1.0),
            # TODO make these as params?
            nn.ReLU(),
            nn_init(nn.Conv2d(24, 24, kernel_size=3, stride=1), w_scale=1.0),
            nn.ReLU(),
            nn_init(nn.Conv2d(24, 48, kernel_size=4, stride=1), w_scale=1.0),
            nn.ReLU(),
        )
        return encoder
