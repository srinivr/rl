from utils.transforms.base_transform import BaseTransform


class StateToFeatureTransform(BaseTransform):

    def __init__(self, feature_agent):
        self.feature_agent = feature_agent

    def transform(self, inputs):
        return self.feature_agent.evaluate(self.feature_agent.get_target_model(), inputs).features
