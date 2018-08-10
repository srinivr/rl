class LinearScheduler:
    def __init__(self, initial_epsilon=1., final_epsilon=0.05, decay_steps=5e4, do_decay=True):
        self.epsilon = initial_epsilon
        self.lower_bound = final_epsilon
        self.decay_steps = decay_steps
        self.decay = (initial_epsilon - final_epsilon) / decay_steps

    def get_epsilon(self):
        return self.epsilon

    def step(self):
        self.epsilon = max(self.lower_bound, self.epsilon - self.decay)

    def get_decay_steps(self):
        return self.decay_steps

    def set_elapsed_steps(self, steps):
        self.epsilon = max(self.lower_bound, self.epsilon - (self.decay_steps * steps))

    def boost(self, rate=100.):
        dist = 1. - self.epsilon
        self.epsilon = max(1., self.epsilon + dist/rate)
