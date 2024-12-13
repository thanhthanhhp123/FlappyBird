import numpy as np
from utils import *
from net import *

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((12, 2))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
            
            if episode % 100 == 0:
                self.save(f"qlearning/flappy_bird_qlearning{episode}.npy")

        self.save("qlearning/flappy_bird_qlearning.npy")

    def save(self, path):
        np.save(path, self.q_table)
    def load(self, path):
        self.q_table = np.load(path)


if __name__ == "__main__":
    import gymnasium
    import flappy_bird_gymnasium
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar = False)
    env.unwrapped.update_constants(**FlappyBirdPresets.easy())
    k = 16
    env = FrameStack(env, k)
    agent = QLearning(env)
    agent.train()
    