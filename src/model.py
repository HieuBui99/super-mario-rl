import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

class Agent(nn.Module):
    def __init__(self, num_frames, num_actions):
        super(Agent, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_frames, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lin = nn.Linear(128 * 6 * 6, 512)

        # Algorithm 1 PPO, Actor-Critic Style
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self, layer):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return self.actor(x), self.critic(x)
    
    def act(self, x):
        logits, state_value = self(x)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), state_value.detach()