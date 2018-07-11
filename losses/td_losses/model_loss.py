from torch import nn

from losses.td_losses.base_loss import BaseLoss


class ModelLoss(BaseLoss):

    def __init__(self, criterion=nn.SmoothL1Loss):
        super().__init__(criterion)

    def get_bootstrap_values(self, model_outputs):
        temp = model_outputs.model_td_predictions.detach()
        return temp

    def get_immediate_values(self, states, actions, rewards):
        return states

    def _get_model_outputs(self, model_outputs, actions):
        return model_outputs.model_td_predictions

    def get_shape(self):
        return [5, 8, 8]

