import random
from collections import deque

class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []