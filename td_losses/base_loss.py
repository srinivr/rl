class BaseLoss:

    def __init__(self, critertion):
        self.criterion = critertion()

    def get_bootstrap_values(self, model_outputs):
        raise NotImplementedError

    def get_immediate_values(self, states, actions, rewards):
        raise NotImplementedError

    def get_loss(self, model_outputs, actions, targets):
        return self.criterion(self._get_model_outputs(model_outputs, actions), targets)

    def _get_model_outputs(self, model_outputs, actions):
        raise NotImplementedError

    def get_shape(self):
        """

        :return: list of dimensions
        """
        raise NotImplementedError
