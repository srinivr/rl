from utils.transforms.input_transforms.state_to_feature_transform import StateToFeatureTransform
from utils.transforms.output_transforms.features_to_q_transform import FeaturesToQTransform


class IterativeAgent:
    def __init__(self, feature_agent, q_agent, n_iters=int(1e6), n_feature_iters=20000, n_q_iters=20000):
        self.feature_agent = feature_agent
        self.q_agent = q_agent
        self.n_iters = n_iters
        self.n_feature_iters = n_feature_iters
        self.n_q_iters = n_q_iters

    def learn(self, feature_env, q_env, feature_eval_env=None, q_eval_env=None, n_eval_episodes=100):
            self.feature_agent.add_output_transform(FeaturesToQTransform(self.q_agent))  # input: transformed states
            self.q_agent.add_input_transform(StateToFeatureTransform(self.feature_agent))
            step_states = None  # TODO how to work with DQN?
            for _ in range(self.n_iters):
                self.feature_agent.learn(feature_env, feature_eval_env, self.n_feature_iters, n_eval_episodes,
                                         step_states)
                self.q_agent.learn(q_env, q_eval_env, self.n_q_iters, n_eval_episodes, step_states)
