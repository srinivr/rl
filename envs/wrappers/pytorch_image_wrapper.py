from gym import Wrapper


#  TODO learn gym wrapper and figure out why it isn't working with subprocvec
class PyTorchImageWrapper(Wrapper):

    def reset(self):
        state = super().reset()
        return self._transform(state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        return self._transform(state), reward, done, info

    @staticmethod
    def _transform(state):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return state.transpose((2, 0, 1))
