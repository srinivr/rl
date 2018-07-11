import torch.nn as nn

from losses.auxiliary_losses.auxiliary_loss import AuxiliaryLoss


class RewardLoss(AuxiliaryLoss):
    def __init__(self, criterion=nn.MSELoss):
        super().__init__(criterion)

    def get_loss(self, model_outputs, actions, rewards, dones):
        return self.criterion(model_outputs.rewards, rewards.view(-1, 1))

    def get_name(self):
        return 'reward_auxiliary_loss'
