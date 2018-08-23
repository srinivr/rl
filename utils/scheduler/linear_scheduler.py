class LinearScheduler:
    def __init__(self, initial_epsilon=1., final_epsilon=0.05, decay_steps=5e4, do_decay=True):
        assert 0. <= initial_epsilon <= 1. and 0. <= final_epsilon <= 1.
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.decay = (initial_epsilon - final_epsilon) / decay_steps

    def get_epsilon(self):
        return self.epsilon

    def step(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.decay)

    def set_elapsed_steps(self, steps):
        self.epsilon = max(self.final_epsilon, self.epsilon - (self.decay_steps * steps))

    def reset(self, initial_epsilon):
        assert 0. <= initial_epsilon <= 1.
        self.epsilon = initial_epsilon
        self.decay = (initial_epsilon - self.final_epsilon) / self.decay_steps

    def get_decay_steps(self):
        return self.decay_steps

    def get_final_epsilon(self):
        return self.final_epsilon
