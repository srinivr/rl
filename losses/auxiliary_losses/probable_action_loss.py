import numpy as np
import torch.nn as nn

from losses.auxiliary_losses.auxiliary_loss import AuxiliaryLoss


class ProbableActionLoss(AuxiliaryLoss):

    def __init__(self, criterion=nn.CrossEntropyLoss, loss_probability=1.):
        super().__init__(criterion)
        self.loss_probability = loss_probability

    def get_loss(self, model_outputs, actions, rewards, dones, auxiliary_info):
        if np.random.random() < self.loss_probability:
            correct_actions = auxiliary_info.actions
            return self.criterion(model_outputs.q_values, correct_actions)
        return 0.0

    def get_name(self):
        return 'action_auxiliary_loss'