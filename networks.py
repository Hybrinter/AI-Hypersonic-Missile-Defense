import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.state_dim = config['env']['state_dim']
        self.action_dim = config['env']['action_dim']
        self.hidden_dim = config['network']['hidden_dim']

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.state_dim = config['env']['state_dim']
        self.action_dim = config['env']['action_dim']
        self.hidden_dim = config['network']['hidden_dim']

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
