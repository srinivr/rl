from losses.auxiliary_losses.auxiliary_loss import AuxiliaryLoss
import torch.nn as nn


class TreeRewardLoss(AuxiliaryLoss):

    def __init__(self, criterion=nn.MSELoss):
        super().__init__(criterion)

    def get_loss(self, model_outputs, actions, rewards, dones=None):
        """
        :param rewards: 1D tensor
        """
        model_rewards = model_outputs.rewards
        return self.criterion(model_rewards.rewards.gather(1, actions.view(-1, 1)), rewards.view(-1, 1))

    def get_name(self):
        return 'tree_reward_loss'

