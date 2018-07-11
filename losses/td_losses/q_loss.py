from torch import nn

from losses.td_losses.base_loss import BaseLoss


class QLoss(BaseLoss):

    def __init__(self, criterion=nn.SmoothL1Loss):
        super().__init__(criterion)

    def get_bootstrap_values(self, model_outputs):
        return model_outputs.q_values.max(1)[0].view(-1, 1).detach()  # max along a dim returns 1D

    def get_immediate_values(self, states, actions, rewards):
        return rewards.view(-1, 1)

    def _get_model_outputs(self, model_outputs, actions):
        return model_outputs.q_values.gather(1, actions.view(-1, 1))

    def get_shape(self):
        return [1]

    def get_name(self):
        return 'q_loss'

