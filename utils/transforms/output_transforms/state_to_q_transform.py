from utils.transforms.output_transforms.base_output_transform import BaseOutputTransform


class StateToQTransform(BaseOutputTransform):

    def __init__(self, f_agent, q_agent):
        self.f_agent = f_agent
        self.q_agent = q_agent

    def transform(self, states, model_outputs):
        f_target_model_outputs = self.f_agent.evaluate(self.f_agent.get_target_model(), states)
        return self.q_agent.evaluate(self.q_agent.get_target_model(), f_target_model_outputs.features)
