import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Agent:
    def __init__(self, actor, critic, index, config):
        self.actor = actor
        self.critic = critic
        self.target_actor = self._create_target_network(actor)
        self.target_critic = self._create_target_network(critic)
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

        self.batch_size = config['training']['batch_size']

        self.gamma = config['agent']['gamma']
        self.tau = config['agent']['tau']

        self.index = index
        self.num_agents = config['pursuers']['num_pursuer_agents'] + config['evaders']['num_evader_agents']

    def _create_target_network(self, network):
        target_network = type(network)(**network.get_init_params())
        target_network.load_state_dict(network.state_dict())
        return target_network

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().cpu().numpy()[0]

    def update(self, agents, sample):
        """

        :param agents: list of agent objects as [pursuers, evaders]
        :param sample: contains a np.array(5, num_agents, batch_size) to represent a set of experiences to train on
        :return:
        """
        states, actions, rewards, next_states, dones = sample

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Update Centralized Critic
        next_actions = torch.zeros(self.num_agents, self.batch_size)
        for index in range(self.num_agents):
            next_actions[index] = agents[index].actor(next_states)

        target_q_values = self.target_critic(next_states, next_actions)
        expected_q_values = rewards + (self.gamma * target_q_values * (1 - dones))
        critic_loss = nn.MSELoss()(self.critic(states, actions), expected_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mse()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)

    def _soft_update(self, source_model, target_model, tau):
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)