import torch.nn as nn

from utils.initializer import nn_init


class Encoder:

    @staticmethod
    def get_push_encoder(n_input_channels=5):
        encoder = nn.Sequential(
            nn_init(nn.Conv2d(in_channels=n_input_channels, out_channels=24, kernel_size=3, stride=1), w_scale=1.0),
            nn.ReLU(),
            nn_init(nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1), w_scale=1.0),
            nn.ReLU(),
            nn_init(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=1), w_scale=1.0),
            nn.ReLU(),
        )
        return encoder
