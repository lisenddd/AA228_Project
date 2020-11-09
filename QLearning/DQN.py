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

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_size = action_size
        self.memory = list()
        self.model = nn.Sequential(
            nn.Linear(state_size[0]*state_size[1], state_size[0]*2),
            nn.ReLU(),
            nn.Linear(state_size[0]*2, self.action_size)
        )
    
    def forward(self, x):
        return self.model(x)
        
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_qvals = self.model(torch.from_numpy(state).float())
        return torch.argmax(action_qvals[0])
    
    def replay(self):
        minibatch = random.sample(self.memory, 15)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.amax(self.model(torch.from_numpy(next_state)).float()[0])
            target_f = self.model(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)



def main():
    env = gym.make('Gomoku9x9-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    gomoku_dqn = DQN(state_size, action_size)
    # Evaluation
    for i in range(1000):
        state = env.reset()
        state = state.flatten()
        for t in range(100):
            action = gomoku_dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            gomoku_dqn.memorize(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/1000, score: {}, e: {:.2}"
                      .format(i, time, agent.epsilon))
                break
            if len(gomoku_dqn.memory) > 40:
                gomoku_dqn.replay()

if __name__ == '__main__':
    main()
