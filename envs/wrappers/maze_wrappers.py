import numpy as np

from gym import Wrapper
from gym_maze.envs.Astar_solver import AstarSolver

class CorrectActionWrapper(Wrapper):

    def __init__(self, env, flush_cache=False):
        super().__init__(env)
        self.cache = dict()
        self.prev_state_correct_action = None
        self._correct_action_called = 0
        self._correct_action_exists = 0
        self.flush_cache = flush_cache

    def step(self, action):
        state, reward, done, info, *auxiliary_info = self.env.step(action)
        correct_action = self.prev_state_correct_action
        self.prev_state_correct_action = None if done else self._get_correct_action(state)
        return (state, reward, done, info, *auxiliary_info, correct_action)

    def reset(self):
        if self.flush_cache:
            self.cache = dict()
        state = super().reset()
        self.prev_state_correct_action = self._get_correct_action(state)
        if self.prev_state_correct_action is None:
            print('The maze is not solvable...creating new maze')
            raise Exception('unsolvable maze')
        return state

    def _get_correct_action(self, state):
        _key = str(state)
        self._correct_action_called += 1
        if _key in self.cache:
            self._correct_action_exists += 1
            return self.cache[_key]
        solver = AstarSolver(self.unwrapped, self.unwrapped.goal_states[0])
        if not solver.solvable():
            return None
        else:
            correct_action = solver.get_actions()[0]
            self.cache[_key] = correct_action
            return correct_action

    def is_solvable(self):
        return self._get_correct_action(self.env.reset()) is not None


class MaxStepWrapper(Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.elapsed_steps = 0

    def step(self, action):
        state, reward, done, info, *auxiliary_info = self.env.step(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_steps:
            done = True
        return (state, reward, done, info, *auxiliary_info)

    def reset(self):
        self.elapsed_steps = 0
        return super().reset()


class FixedRandomEnvsWrapper(Wrapper):
    def __init__(self, envs):
        self.envs = envs
        self.env = np.random.choice(self.envs, 1)[0]
        super().__init__(self.env)

    def reset(self):
        self.env = np.random.choice(self.envs, 1)[0]
        return self.env.reset()

    def step(self, action):
        state, reward, done, info, *auxiliary_info = super().step(action)
        return (state, reward, done, info, *auxiliary_info)

