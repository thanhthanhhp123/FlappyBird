
import tqdm
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


class DeepQLearning():
    def __init__(self, env, lr = 1e-4, gamma = 0.9, batch_size = 64, device = 'cpu', k = 4):
        self.state_size = 12
        self.action_size = 2
        self.memory = Memory()
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.device = device

        self.q_network = DQNNetwork(self.state_size, self.action_size, k = k).to(device)
        self.target_network = DQNNetwork(self.state_size, self.action_size, k = k).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
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

        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        target_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = self.loss_fn(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state, epsilon):
        if torch.rand(1).item() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.q_network.select_action(state)
        else:
            return torch.randint(0, self.action_size, (1,)).item()
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
    
    def test(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.q_network.select_action(state)
        
def train():
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar = False)
    env.unwrapped.update_constants(**FlappyBirdPresets.easy())
    k = 16
    env = FrameStack(env, k)

    writer = SummaryWriter()
    agent = DeepQLearning(env, device = device, k = k)
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
            agent.save(f"models/flappy_bird_{episode}_episode.pth")
        if episode % 10 == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.save("models/flappy_bird_last.pth")


def test():
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env.unwrapped.update_constants(**FlappyBirdPresets.easy())
    k = 16
    env = FrameStack(env, k)
    agent = DeepQLearning(env, device = device, k = k)
    agent.load(r"models\flappy_bird_122300_episode.pth")
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.test(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

if __name__ == "__main__":
    test()





