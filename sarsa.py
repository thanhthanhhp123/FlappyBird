import numpy as np
import random

class Sarsa:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state][action]
        target = reward + self.gamma * self.q_table[next_state][next_action]
        self.q_table[state][action] += self.alpha * (target - predict)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.learn(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
