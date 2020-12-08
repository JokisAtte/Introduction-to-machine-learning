# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time


def method():
    # Environment
    env = gym.make("Taxi-v3")
    # Training parameters for Q learning
    alpha = 0.9 # Learning rate
    gamma = 0.9 # Future reward discount factor
    #num_of_episodes = 1000
    num_of_episodes = 1000000
    #num_of_steps = 500 # per each episode
    num_of_steps = 500
    # Q tables for rewards
    # Q_reward = -100000*np.ones((500,6))
    Q_reward = np.zeros((500,6))

    # Training w/ random sampling of actions
    # YOU WRITE YOUR CODE HERE
    #Exploration params
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01

    rewards = []
    num_of_actions = []
    #for episodes for steps
    for episode in range(num_of_episodes):
        state = env.reset()
        tot_reward = 0

        for step in range(num_of_steps):
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q_reward[state, :])
            else:
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            tot_reward += reward
            Q_reward[state, action] = Q_reward[state, action] + alpha * (reward + gamma * np.max(Q_reward[new_state, :])
                                                                        - Q_reward[state, action])
            env.render()
            # time.sleep(1)
            state = new_state
            if done:
                rewards.append(tot_reward)
                num_of_actions.append(step)
                print("Total reward %d" % tot_reward)
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    print("avg total reward:", np.mean(rewards))
    print("avg number of steps", np.mean(num_of_actions))

def main():
    method()

main()