import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Agent:
    def __init__(self, actor, critic, replay_buffer, config):
        self.actor = actor
        self.critic = critic
        self.actor_target = self._create_target_network(actor)
        self.critic_target = self._create_target_network(critic)
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

        self.replay_buffer = replay_buffer
        self.max_replay_buffer_len = config['agent']['batch_size'] * config['training']['max_episode_len']

        self.gamma = config['agent']['gamma']
        self.tau = config['agent']['tau']

    def _create_target_network(self, network):
        target_network = type(network)(*network.args, **network.kwargs)
        target_network.load_state_dict(network.state_dict())
        return target_network

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().cpu().numpy()[0]

    def update(self, batch_size):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Update Critic
        next_actions = self.actor(next_states)
        target_q_values = self.critic(next_states, next_actions)
        expected_q_values = rewards + (self.gamma * target_q_values * (1 - dones))
        critic_loss = nn.MSELoss()(self.critic(states, actions), expected_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

    def soft_update(self, source_model, target_model, tau):
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
