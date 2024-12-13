import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, k = 4):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size * k, 128) 
        self.fc2 = nn.Linear(128, 256)      
        self.fc3 = nn.Linear(256, action_size) 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        return self.fc3(x)          
    
    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action = Q.argmax(dim=1)
        return action.item()
    
class DuelDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, k = 4):
        super(DuelDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size * k, 128) 
        self.fc2 = nn.Linear(128, 256)      
        self.fc3 = nn.Linear(256, 1) 
        self.fc4 = nn.Linear(256, action_size) 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        advantage = self.fc4(x)
        Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return Q
    
    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action = Q.argmax(dim=1)
        return action.item()