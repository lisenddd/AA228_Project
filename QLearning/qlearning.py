import numpy as np
import copy
import gym
from gym_gomoku.envs import util
from gym_gomoku.envs import gomoku

def train(num_episode, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
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
    return q_table


def main():
    env = gym.make('Gomoku7x7-v0')
    num_episode = 10000
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    q_table = train(num_episode, alpha, gamma, epsilon)

    total_epochs = 0
    total_penalties = 0
    eval_episodes = 100

    # Evaluation
    for i in range(eval_episodes):
        _, state = env.reset()
        epochs = 0
        penalties = 0
        reward = 0
        terminated = False

        while not terminated:
            env.render('human')
            action = np.argmax(q_table[state])
            state, reward, terminated, info = environment.step(action)

            if reward == -1:
                penalties += 1
            
            epochs += 1

        total_epochs += epochs
        total_penalties += penalties
    
    print("Results")
    print("**********************************")
    print("Epochs per episode: {}".format(total_epochs / eval_episodes))
    print("Penalties per episode: {}".format(total_penalties / eval_episodes))

if __name__ == '__main__':
    main()
