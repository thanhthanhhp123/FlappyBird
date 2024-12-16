import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import *
from net import *

import gymnasium
import flappy_bird_gymnasium


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleQLearning():
    def __init__(self, env, lr = 1e-4, gamma = 0.9, batch_size = 64, device = 'cpu', k = 4):
        self.state_size = 12
        self.action_size = 2
        self.memory = Memory()
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.device = device

        if not os.path.exists("double"):
            os.mkdir("double")

        self.q_network1 = DQNNetwork(self.state_size, self.action_size, k = k).to(device)
        self.q_network2 = DQNNetwork(self.state_size, self.action_size, k = k).to(device)
        self.target_network1 = DQNNetwork(self.state_size, self.action_size, k = k).to(device)
        self.target_network2 = DQNNetwork(self.state_size, self.action_size, k = k).to(device)
        self.target_network1.load_state_dict(self.q_network1.state_dict())
        self.target_network2.load_state_dict(self.q_network2.state_dict())

        self.target_network1.eval()
        self.target_network2.eval()

        self.optimizer1 = optim.Adam(self.q_network1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.q_network2.parameters(), lr=lr)

        self.loss_fn = nn.SmoothL1Loss()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values1 = self.q_network1(states)
        q_values2 = self.q_network2(states)

        next_q_values1 = self.target_network1(next_states)
        next_q_values2 = self.target_network2(next_states)

        next_actions = self.q_network1(next_states).argmax(1)

        next_q_value = next_q_values2.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        q_value1 = q_values1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_value2 = q_values2.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss1 = self.loss_fn(q_value1, target_q_value.detach())
        loss2 = self.loss_fn(q_value2, target_q_value.detach())

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()




    def act(self, state, epsilon):
        if torch.rand(1).item() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.q_network1.select_action(state)
        else:
            return torch.randint(0, self.action_size, (1,)).item()
        
    def save(self, path):
        torch.save(self.q_network1.state_dict(), path)

    def load(self, path):
        self.q_network1.load_state_dict(torch.load(path))

    def test(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.q_network1.select_action(state)
        
def train():
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar = False)
    env.unwrapped.update_constants(**FlappyBirdPresets.easy())
    k = 16
    env = FrameStack(env, k)

    writer = SummaryWriter()
    agent = DoubleQLearning(env, device = device, k = k)
    agent.load(r"models\flappy_bird_121200_episode.pth")
    epsilon = 1
    epsilon_decay = 0.999
    epsilon_min = 0.01
    episodes = 1000000
    pbar = tqdm.tqdm(range(121200, episodes), desc="Episodes")
    for episode in pbar:
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state.flatten(), epsilon)
            next_state, reward, done, _, info = env.step(action)
            agent.memory.add((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
        pbar.set_postfix({"Reward": total_reward, "Epsilon": epsilon})

        writer.add_scalar("Reward", total_reward, episode)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if episode % 100 == 0:
            agent.save(f"double/flappy_bird_{episode}_episode.pth")
        if episode % 10 == 0:
            agent.target_network1.load_state_dict(agent.q_network1.state_dict())
            agent.target_network2.load_state_dict(agent.q_network2.state_dict())
    agent.save("double/flappy_bird_last.pth")

def test():
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env.unwrapped.update_constants(**FlappyBirdPresets.easy())
    k = 16
    env = FrameStack(env, k)
    agent = DoubleQLearning(env, device = device, k = k)
    agent.load(r"models\flappy_bird_122300_episode.pth")
    state, _ = env.reset()
    while True:
        action = agent.test(state.flatten())
        state, _, done, _, _ = env.step(action)
        if done:
            break

if __name__ == "__main__":
    train()
    test()