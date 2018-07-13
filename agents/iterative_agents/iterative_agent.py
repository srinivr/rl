from utils.transforms.input_transforms.state_to_feature_transform import StateToFeatureTransform
from utils.transforms.output_transforms.feature_to_q_transform import FeatureToQTransform
from utils.transforms.output_transforms.state_to_q_transform import StateToQTransform


class IterativeAgent:
    def __init__(self, feature_agent, q_agent, n_iters=int(1e6), n_feature_iters=80, n_q_iters=80):
        self.feature_agent = feature_agent
        self.q_agent = q_agent
        self.n_iters = n_iters
        self.n_feature_iters = n_feature_iters
        self.n_q_iters = n_q_iters

    def learn(self, feature_env, q_env, feature_eval_env=None, q_eval_env=None, n_eval_episodes=100):
            self.feature_agent.add_output_transform(StateToQTransform(self.feature_agent, self.q_agent))
            #self.feature_agent.add_output_transform(FeaturesToQTransform(self.q_agent))  # input: transformed states
            self.q_agent.add_input_transform(StateToFeatureTransform(self.feature_agent))
            f_step_states, f_rewards, f_lengths = None, None, None  # TODO how to work with DQN?
            q_step_states, q_rewards, q_lengths = None, None, None
            for _ in range(self.n_iters):
                f_step_states, f_rewards, f_lengths = self.feature_agent.learn(feature_env, feature_eval_env,
                                                                               self.n_feature_iters, n_eval_episodes,
                                                                               f_step_states, f_rewards, f_lengths)
                q_step_states, q_rewards, q_lengths = self.q_agent.learn(q_env, q_eval_env, self.n_q_iters,
                                                                         n_eval_episodes, q_step_states, q_rewards,
                                                                         q_lengths)
