import torch.nn as nn

from utils.initializer import nn_init


class Decoder:

    @staticmethod
    def get_push_decoder(n_input_channels=48):
        decoder = nn.Sequential(
            nn_init(nn.ConvTranspose2d(n_input_channels, 48, 4)),
            nn.ReLU(),
            nn_init(nn.ConvTranspose2d(48, 24, 3)),
            nn.ReLU(),
            nn_init(nn.ConvTranspose2d(24, 5, 3))
        )
        return decoder
