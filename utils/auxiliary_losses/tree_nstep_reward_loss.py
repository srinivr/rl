from utils.auxiliary_losses.auxiliary_loss import AuxiliaryLoss


class TreeNStepRewardLoss(AuxiliaryLoss):

    def __init__(self, depth, n_step):
        super().__init()
        self.depth = depth
        self.n_step = n_step

    def get_batch(self, model_outputs, actions, rewards, done):
        q_values_len = model_outputs.q_values.size()[0]
        model_rewards = model_outputs.rewards
        outputs = []
        targets = []
        for i in range(q_values_len):
            outputs.append()
            pass
        return outputs, targets
