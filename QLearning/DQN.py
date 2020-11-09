import numpy as np
import copy
import gym
import random
from gym_gomoku.envs import util
from gym_gomoku.envs import gomoku

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, state_size):
        super(DQN, self).__init__()
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.1
        self.num_action = state_size[0] * state_size[1]
        self.model = nn.Sequential(
            nn.Linear(state_size[0]*state_size[1], state_size[0]*2),
            nn.ReLU(),
            nn.Linear(state_size[0]*2, self.num_action)
        )
    
    def forward(self, x):
        return self.model(x)
        
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_action)
        action_qvals = self.model.predict(state)
        return np.argmax(action_qvals[0])
    
    def replay(self):
        minibatch = random.sample(self.memory, 10)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.modek.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
