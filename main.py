from environment import PursuerEvaderEnv
from agent import Agent
from networks import Actor, CentralizedCritic
from replay_buffer import ReplayBuffer
from config_manager import ConfigManager
import numpy as np


def train(env, pursuers, evaders, config):
    batch_size = config['agent']['batch_size']
    num_episodes = config['training']['num_episodes']

    for episode in range(config['training']['num_episodes']):
        states = env.reset()
        total_reward = 0

        while True:
            actions = [agent.select_action(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones, _ = env.step(actions)
            total_reward += sum(rewards)

            for i, agent in enumerate(agents):
                agent.replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states

            if all(dones):
                break

            for agent in agents:
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager(config_dir='configs')
    config_manager.load_configs()

    config = config_manager.get_config('initial_config')

    env = PursuerEvaderEnv(config)

    pursuers = []
    for index in range(config['pursuers']['num_pursuer_agents']):
        actor = Actor(config)
        critic = CentralizedCritic(config)
        replay_buffer = ReplayBuffer(config)
        agent = Agent(actor, critic, index, replay_buffer, config)
        pursuers.append(agent)

    evaders = []
    for index in range(config['evaders']['num_evader_agents']):
        actor = Actor(config)
        critic = CentralizedCritic(config)
        replay_buffer = ReplayBuffer(config)
        agent = Agent(actor, critic, index, replay_buffer, config)
        evaders.append(agent)

    train(env, pursuers, evaders, config)

