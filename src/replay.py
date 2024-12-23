import random
from collections import deque

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=100000)
    def add(self, timestep):
        self.buffer.append(timestep)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)