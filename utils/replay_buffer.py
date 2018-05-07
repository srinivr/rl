import random


class ReplayBuffer:

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.len = 0

    def insert(self, x):
        if self.len == self.size:
            del self.buffer[0]
        else:
            self.len += 1
        self.buffer.append(x)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return self.len

    def __str__(self):
        return str(self.buffer)

