from utils.transforms.output_transforms.base_output_transform import BaseOutputTransform


class FeaturesToQTransform(BaseOutputTransform):

    def __init__(self, q_agent):
        self.q_agent = q_agent

    def transform(self, states, model_outputs):
        return self.q_agent.evaluate(self.q_agent.get_target_model(), model_outputs.features)
