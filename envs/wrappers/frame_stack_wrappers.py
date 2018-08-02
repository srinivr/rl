import numpy as np

from gym import Wrapper


class MultiProcessFrameStackWrapper(Wrapper):

    def __init__(self, vec_env, num_envs, n_stack=4):
        if not hasattr(vec_env, 'reward_range'):
            vec_env.reward_range = None  # just because wrappers use this
            vec_env.metadata = None  # just because wrappers use this
        super().__init__(vec_env)
        self.n_stack = n_stack
        self.obs = np.zeros((num_envs, n_stack, *vec_env.reset().shape[2:]))  # buffer

    def step(self, action):
        batch_states, batch_rewards, batch_dones, batch_info = super().step(action)
        self.obs = np.roll(self.obs, -1, axis=1)
        self.obs[:, -1, :, :] = batch_states[0]
        batch_states = np.copy(self.obs)
        for i in range(self.env.num_envs):
            if batch_dones[i]:
                self.obs[i] *= 0.
        return batch_states, batch_rewards, batch_dones, batch_info

    def reset(self, **kwargs):
        batch_states = super().reset()
        self.obs *= 0.
        self.obs[:, -1, :, :] = batch_states[0]
        return np.copy(self.obs)


class FrameStackWrapper(MultiProcessFrameStackWrapper):

    def step(self, action):
        state, reward, done, info = super().step([action])
        return state[0], reward[0], done[0], info[0]

    def reset(self):
        return super().reset()[0]
