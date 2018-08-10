class DecayScheduler:
    def __init__(self, initial_epsilon=1., lower_bound=0., decay=0.99988, do_decay=True):
        self.epsilon = initial_epsilon
        self.lower_bound = lower_bound
        self.decay = decay
        self.do_decay = do_decay

    def get_epsilon(self, n=None):
        return self.epsilon

    def step(self):
        self.epsilon = max(self.lower_bound, self.decay * self.epsilon)

    def set_elapsed_steps(self, steps):
        raise NotImplementedError
