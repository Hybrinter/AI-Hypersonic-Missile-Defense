import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


class PursuerEvaderEnv:
    def __init__(self, config):
        self.grid_dim = config['env']['grid_dim']
        self.num_pursuers = config['pursuers']['num_pursuers']
        self.num_evaders = config['evaders']['num_evaders']

        self.pursuer_start = np.array(config['pursuers']['pursuer_start'])
        self.evader_start = np.array(config['evaders']['evader_start'])
        self.catch_range = config['pursuers']['catch_range']

        self.pursuer_position = self.pursuer_start
        self.evader_position = self.evader_start

        self.pursuer_speed = config['pursuers']['pursuer_speed']
        self.evader_speed = config['evaders']['evader_speed']

        self.done = False
        self.reset()

    def reset(self):
        self.pursuer_position = self.pursuer_start
        self.evader_position = self.evader_start
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return {
            'pursuer': self.pursuer_position.copy(),
            'evader': self.evader_position.copy()
        }

    def step(self, pursuer_action, evader_action):
        if not self.done:
            print(f'Pursuer Position: {self.pursuer_position}')
            print(f'Evader Position: {self.evader_position}')
            self._move(self.pursuer_position, pursuer_action, True)
            self._move(self.evader_position, evader_action, False)

            if self._is_caught():
                reward = 1
                self.done = True
            elif self._reached_territory():
                reward = -1
                self.done = True
            else:
                reward = 0

            return (self._get_obs(), reward, self.done,
                    {'distance': np.linalg.norm(self.pursuer_position - self.evader_position)})
        else:
            raise ValueError("Game is done. Please reset the environment.")

    def _move(self, pos, action, is_pursuer):
        speed = self.pursuer_speed if is_pursuer else self.evader_speed
        direction_norm = 1 / math.sqrt(action[0] ** 2 + action[1] ** 2)

        if action[0] < 0:  # left
            pos[0] = max(round(pos[0] + action[0] * speed * direction_norm), 0)
        elif action[0] > 0:  # right
            pos[0] = min(round(pos[0] + action[0] * speed * direction_norm), self.grid_dim[0] - 1)

        if action[1] < 0:  # down
            pos[1] = max(round(pos[1] + action[1] * speed * direction_norm), 0)
        elif action[1] > 0:  # up
            pos[1] = min(round(pos[1] + action[1] * speed * direction_norm), self.grid_dim[1] - 1)

    def _is_caught(self):
        return np.linalg.norm(self.pursuer_position - self.evader_position) <= self.catch_range

    def _reached_territory(self):
        return self.evader_position[0] == 0

    def render(self):
        grid = np.zeros((self.grid_dim[0], self.grid_dim[1], 3))
        grid[self.grid_dim[1] - self.pursuer_position[1], self.pursuer_position[0]] = [255, 0, 0]  # Red for pursuer
        grid[self.grid_dim[1] - self.evader_position[1], self.evader_position[0] - 1] = [0, 0, 255]  # Blue for evader

        fig, ax = plt.subplots()
        ax.imshow(grid)

        # Draw circle around pursuer
        circle_pursuer = patches.Circle(
            (float(self.pursuer_position[0]), float(self.grid_dim[1] - self.pursuer_position[1])),
            self.catch_range, fill=False, edgecolor='red', linestyle='dashed'
        )
        ax.add_patch(circle_pursuer)

        plt.axis('off')
        plt.show()
