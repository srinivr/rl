import numpy as np
import torch.nn as nn

from losses.auxiliary_losses.auxiliary_loss import AuxiliaryLoss


class SamplesActionLoss(AuxiliaryLoss):

    def __init__(self, criterion=nn.CrossEntropyLoss, n_samples=None):
        super().__init__(criterion)
        self.n_samples = n_samples

    def get_loss(self, model_outputs, actions, rewards, dones, auxiliary_info):
        # number of samples to compute loss
        n_samples = self.n_samples if self.n_samples is not None else actions.size()[0]
        assert n_samples <= actions.size()[0]

        correct_actions = auxiliary_info.actions
        if n_samples == 0:
            return 0.
        indices = np.random.choice(correct_actions.size()[0], n_samples, replace=False)
        return self.criterion(model_outputs.q_values[indices], correct_actions[indices])

    def get_name(self):
        return 'action_auxiliary_loss'