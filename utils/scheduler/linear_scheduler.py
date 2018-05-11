class LinearScheduler:
    def __init__(self, initial_epsilon=1., lower_bound=0.05, decay_steps=5e5, do_decay=True):
        self.epsilon = initial_epsilon
        self.lower_bound = lower_bound
        self.decay = (initial_epsilon - lower_bound) / decay_steps
        self.do_decay = do_decay
        # TODO log values

    def get_epsilon(self, n=None):
        return self.epsilon

    def step(self):
        if self.do_decay:
            # TODO log values
            self.epsilon = max(self.lower_bound, self.epsilon - self.decay)

    def set_no_decay(self):
        # TODO log values
        self.do_decay = False

    def set_do_decay(self):
        # TODO log values
        self.do_decay = True
