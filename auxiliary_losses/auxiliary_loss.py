class AuxiliaryLoss:

    def __init__(self, loss_criterion):
        self.criterion = loss_criterion

    def get_loss(self, model_outputs, actions, rewards):
        """
        inputs must by pytorch tensors
        """
        raise NotImplementedError