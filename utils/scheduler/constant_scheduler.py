class StepDecayScheduler:
    def __init__(self, epsilon=1.):
        self.epsilon = epsilon

    def get_epsilon(self, n):
        return self.epsilon

    def step(self):
        self.epsilon *= 0.995 # decay epsilon every step
