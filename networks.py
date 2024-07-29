import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.dim = len(config['env']['grid_size'])
        self.layer_size = config['network']['layer_size']
        self.num_agents = config['pursuers']['num_pursuer_agents'] + config['evaders']['num_evader_agents']
        self.batch_size = config['training']['batch_size']

        self.fc1 = nn.Linear(self.dim * self.num_agents, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, self.dim)

    def forward(self, state):
        """
        :param state: state=(position, agents, batch) with tensor(dim, num-agents, batch_size) and batch_size=1 for eval()
        :return: action=(position, batch) with tensor(dim, batch_size) with a range of [-1,1]
        """
        # flatten position vector
        state = state.view(self.dim * self.num_agents, self.batch_size).transpose(0, 1)

        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.transpose(0, 1).view(self.dim, self.batch_size)


class CentralizedCritic(nn.Module):
    def __init__(self, config):
        super(CentralizedCritic, self).__init__()
        self.num_agents = config['pursuers']['num_pursuer_agents'] + config['evaders']['num_evader_agents']
        self.layer_size = config['network']['layer_size']

        self.fc1 = nn.Linear(2 * self.num_agents, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, 1)

    def forward(self, agent_states, agent_actions):
        """

        :param agent_states: agent_state=(x, y, agents, batch) with tensor(1, 1, num_agents, batch_size)
        :param agent_actions:
        :return:
        """
        x = torch.cat([agent_states, agent_actions], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
