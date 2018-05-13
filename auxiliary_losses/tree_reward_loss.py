from utils.auxiliary_losses.auxiliary_loss import AuxiliaryLoss
import torch.nn as nn


class TreeRewardLoss(AuxiliaryLoss):

    def __init__(self, loss_criterion=nn.MSELoss()):
        super().__init__(loss_criterion)

    def get_loss(self, model_outputs, actions, rewards):
        """

        :param model_outputs:
        :param actions:
        :param rewards: 1D tensor
        :return:
        """
        model_rewards = model_outputs.rewards
        return self.criterion(model_rewards.rewards.gather(1, actions.view(-1, 1)), rewards.view(-1, 1))
