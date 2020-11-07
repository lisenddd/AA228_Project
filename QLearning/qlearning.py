import numpy as np
import copy
import gym
from gym_gomoku.envs import utiol
from gym_gomoku.envs import gomoku

def train(num_episode, alpha, gamma, epsilon):
    for i in range(num_episode):
        state = env.reset()
        reward = 0
        terminated = False

        while not terminated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
        next_state, reward, terminated = env.step(action)
        q_value = q_tablep[state, action]
        max_val = np.max(q_table[next_state])
        new_q_value = (1-alpha) * q_value + alpha * (reward + gamma * max_val)
        q_table[state, action] = new_q_value
        state = next_state
    
    if (episode+1) % 100 == 0:
        env.render()

def evaluate(state, num_episode, epoch, penalty, reward):
    terminated = False
    while not terminated:
        action = np.argmax(q_table[state])
        state, reward, terminated, info = enviroment.step(action)
        if reward < 0:
            penalty += 1
        epoch += 1


def main():
    env = gym.make('Gomoku7x7-v0')
    _, state = env.reset()
    num_episode = 1000
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    train(num_episode, alpha, gamma, epsilon)
    done = False
    while not done:
        env.render('human')
        

if __name__ == '__main__':
    main()
