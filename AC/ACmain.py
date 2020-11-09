# ##### self play #####
# import gym
# import gym_gomoku
# from agent import Agent
# import wandb

# def flipstate(s,a,statedim):
#     news = -s.copy()
#     news[news==-1] = 2
#     news[news==-2] = 1
#     coord = action_to_coord(a,statedim)
#     news[coord[0],coord[1]] = 2
#     return news

# def action_to_coord(a,statedim):
#     coord = (a // statedim, a % statedim)
#     return coord

# def verify_performance(agent,valwin,episode):
#     env = gym.make('Gomoku9x9random-v0')
#     s = env.reset()
#     totalr = 0
#     for steps in range(100):
#         a = agent.take_action(s)
#         sp, r, done, info = env.step(a)
#         totalr += r
#         s = sp
#         if done:
#             break
#     if r > 0:
#         valwin += 1
#     wandb.log({"Validdation Reward": totalr,'Validation Win':valwin,'Validation steps':steps},step=episode)
#     return valwin

# run = wandb.init(project="aa228final",reinit=True,name="test9x9")
# env = gym.make('Gomoku9x9duo-v0') # default 'beginner' level opponent policy

# s = env.reset()
# env.render('ansi')
# # env.step(15) # place a single stone, black color first

# statedim = env.observation_space.shape[0]

# agent = Agent(statedim**2,statedim**2,0.0003,0.95,device="cuda:0")
# agent2 = Agent(statedim**2,statedim**2,0.0003,0.95,device="cuda:0")

# # play a game
# win,win2,valwin = 0,0,0
# for episode in range(20000):
#     totalr,totalr2 = 0,0
#     s = env.reset()
#     for steps in range(100):
#         # Black go first
#         a = agent.take_action(s)
#         # action = env.action_space.sample() # sample without replacement

#         # Self play is white 
#         s2 = flipstate(s,a,statedim)  # flip 1 and 2 in state
#         a2 = agent2.take_action(s2)
#         env.oppoact = a2

#         sp, r, done, info = env.step(a)

#         sp2,r2 = flipstate(sp,a,statedim),-r
#         # r+=0.02
#         loss = agent.train(s,r,sp,done)
#         loss2 = agent2.train(s2,r2,sp2,done)
#         s,s2 = sp,sp2
#         totalr += r
#         totalr2 += r2
#         if episode%10 == 0:
#             # env.render('human')
#             pass
#         else:
#             # env.render('ansi')
#             pass
            
#         if done:
#             # print ("Game is Over")
#             break
#     if episode%10 == 0:
#         valwin = verify_performance(agent,valwin,episode)

#     if r > 0:
#         win += 1
#     if r2 > 0:
#         win2 += 1
#     wandb.log({"total reward": totalr,'Loss':loss,'win':win,'steps this episode':steps},step=episode)
#     wandb.log({"total reward2": totalr2,'Loss2':loss2,'win2':win2},step=episode)
#     print(episode,totalr,totalr2)
# run.finish()

# ##### Selfplay 2 #####
import gym
import gym_gomoku
from agent import Agent
import wandb
import numpy as np

def flipstate(s,a,statedim):
    news = -s.copy()
    news[news==-1] = 2
    news[news==-2] = 1
    coord = action_to_coord(a,statedim)
    news[coord[0],coord[1]] = 2
    return news

def action_to_coord(a,statedim):
    coord = (a // statedim, a % statedim)
    return coord

def verify_performance(agent,valwin,episode):
    agent.saveNetwork()
    if episode < 35000:
        env = gym.make('Gomoku9x9random-v0')
    else:
        env = gym.make('Gomoku9x9-v0')
    s,_ = env.reset()
    totalr = 0
    for steps in range(100):
        a = agent.take_action(s)
        sp, r, done, info = env.step(a)
        totalr += r
        s = sp
        if done:
            break
    if r > 0:
        valwin += 1
    wandb.log({"Validdation Reward": totalr,'Validation Win':valwin,'Validation steps':steps},step=episode)
    return valwin

run = wandb.init(project="aa228final",reinit=True,name="test9x9")
env = gym.make('Gomoku9x9duo-v0') # default 'beginner' level opponent policy

s,_ = env.reset()
env.render('ansi')
# env.step(15) # place a single stone, black color first

statedim = env.observation_space.shape[0]

agent = Agent(statedim**2,statedim**2,0.0003,0.95,device="cuda:0",selfplay=True)

# play a game
win,valwin = 0,0
winlist = []
for episode in range(200000):
    totalr = 0
    s,_ = env.reset()
    for steps in range(100):
        a = agent.take_action(s)

        # Self play is white 
        s2 = flipstate(s,a,statedim)  # flip 1 and 2 in state
        a2 = agent.take_action2(s2)
        env.oppoact = a2

        sp, r, done, info = env.step(a)
        # r+=0.02
        loss = agent.train(s,r,sp,done)
        s = sp
        totalr += r
        if episode%10 == 0:
            # env.render('human')
            pass
        else:
            # env.render('ansi')
            pass
            
        if done:
            # print ("Game is Over")
            break

    if episode%25 == 0:
        valwin = verify_performance(agent,valwin,episode)
    if r > 0:
        win += 1
        winlist.append(1)
    else:
        winlist.append(0)
    
    if len(winlist)>100:
        winlist.pop(0)
        if np.mean(np.array(winlist))>0.65:
            agent.update_target(agent.actorcritic_targ,agent.actorcritic)
            winlist = []

    wandb.log({"total reward": totalr,'Loss':loss,'win':win,'winrate':np.mean(np.array(winlist)),'steps this episode':steps},step=episode)
    print(episode,totalr)
run.finish()


#### vanilla AC #####
# import gym
# import gym_gomoku
# from agent import Agent
# import wandb

# run = wandb.init(project="aa228final",reinit=True,name="test9x9")
# env = gym.make('Gomoku9x9-v0') # default 'beginner' level opponent policy

# s = env.reset()
# env.render('ansi')
# # env.step(15) # place a single stone, black color first

# statedim = env.observation_space.shape[0]

# agent = Agent(statedim**2,statedim**2,0.0003,0.95,device="cuda:0")

# # play a game
# win = 0
# for episode in range(200000):
#     totalr = 0
#     s = env.reset()
#     for steps in range(100):
#         a = agent.take_action(s)
#         sp, r, done, info = env.step(a)
#         # r+=0.02
#         loss = agent.train(s,r,sp,done)
#         s = sp
#         totalr += r
#         if episode%10 == 0:
#             # env.render('human')
#             pass
#         else:
#             # env.render('ansi')
#             pass
            
#         if done:
#             # print ("Game is Over")
#             break

#     if r > 0:
#         win += 1
#     wandb.log({"total reward": totalr,'Loss':loss,'win':win,'steps this episode':steps},step=episode)
#     print(episode,totalr)
# run.finish()

##### Cartpole test #####
# import gym
# import gym_gomoku
# from agent import Agent
# import wandb
# env = gym.make('CartPole-v0') # default 'beginner' level opponent policy
# run = wandb.init(project="aa228final",reinit=True,name="cartpole")
# s = env.reset()

# actiondim = env.action_space.n
# statedim = env.observation_space.shape[0]

# agent = Agent(statedim,actiondim,0.001,0.99)

# # play a game
# for episode in range(5000):
#     totalr = 0
#     s = env.reset()
#     for _ in range(500):
#         a = agent.takeaction(s)
#         action = env.action_space.sample() # sample without replacement
#         sp, r, done, info = env.step(a)
#         # r+=0.1
#         loss = agent.train(s,r,sp,done)
#         s = sp
#         totalr += r
#         # if episode%100 == 0:
#         #     env.render('human')
#         # else:
#         #     env.render('ansi')
            
#         if done:
#             # print ("Game is Over")
#             break

#     print(episode,totalr)
#     wandb.log({"total reward": totalr,'Loss':loss},step=episode)

# run.finish()