# ##### Selfplay 2 #####
import gym
import gym_gomoku
from agent import Agent
import wandb
import numpy as np

env = gym.make('Gomoku9x9-v0') # default 'beginner' level opponent policy

s = env.reset()
env.render('ansi')

statedim = env.observation_space.shape[0]
agent = Agent(statedim**2,statedim**2,0.0003,0.95,device="cuda:0",selfplay=True,loadold=True)

# play a game
for episode in range(2):
    totalr = 0
    s = env.reset()
    for steps in range(100):
        a = agent.take_action(s)

        sp, r, done, info = env.step(a)
        s = sp
        totalr += r
        env.render('human')

            
        if done:
            # print ("Game is Over")
            break
    print(episode,totalr)
