import torch
import torch.nn.functional as F

from models.base_model import BaseModel


class TreeQNModel(BaseModel):
    def __init__(self, tuple_attributes):
        super().__init__(tuple_attributes)

    def forward(self, *input):
        pass

    @staticmethod
    def get_input_dimension():
        pass

    def _backup_fn(self, q_a):
        return torch.sum(F.softmax(q_a, dim=1) * q_a, dim=1, keepdim=True)

    def _q_a(self, s, depth):  # return dim: b x a
        """
        perform action independent + dependent transition and compute the backup
        :param s: normalized_state embedding
        """
        reward = self.reward_fn(s)
        s = s + self.action_independent_transition(s)  # TODO cleaner placement of action_independent_transition
        v_n = []
        transition_rewards = []
        for idx in range(self.n_actions):
            transition = s + self.action_transition[idx](s)  # residual connection
            transition = transition / transition.norm(dim=1, keepdim=True)  # normalize the transition state
            v, transition_return = self._v_fn(transition, depth + 1)
            v_n.append(v)
            transition_rewards.append(transition_return)
        v_n = torch.cat(v_n, dim=1)
        return reward + self.gamma * v_n, self.reward_tuple(reward, transition_rewards)

    def _v_fn(self, s, depth):
        """
        :param s: normalized s
        :return: 1 x 1
        """
        if depth == self.depth:
            return self.value_fn(s), None
        action_values, transition_rewards = self._q_a(s, depth)
        backup = self._backup_fn(action_values)  # assert shape is b x 1
        return (1. - self.lambd) * self.value_fn(s) + self.lambd * backup, transition_rewards
