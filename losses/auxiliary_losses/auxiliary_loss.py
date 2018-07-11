class AuxiliaryLoss:

    def __init__(self, criterion):
        self.criterion = criterion()

    def get_loss(self, model_outputs, actions, rewards, dones):
        """
        inputs must by pytorch tensors
        """
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError
