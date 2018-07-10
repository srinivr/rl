from utils.transforms.base_transform import BaseTransform


class FeaturesToQTransform(BaseTransform):

    def __init__(self, q_agent):
        self.q_agent = q_agent

    def transform(self, inputs):
        return self.q_agent.evaluate(self.q_agent.get_target_model(), inputs.features)
