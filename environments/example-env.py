from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

MAP = [
    "+-----------+",
    "|R: | : : |G|",
    "| : | : : | |",
    "| : : : : | |",
    "| | : | | : |",
    "| | : | | : |",
    "|Y| | : :B| |"
    "+-----------+",
]

class ExampleEnv(Env):
    metadata = {
        "render_modes": ["ansi", "rgb_array"],
        "render_fps": 4 
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 5), (5, 0), (5, 4)]
        self.locs_colours = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        num_states = 720
        num_rows = 6
        num_cols = 6
        max_row = num_rows - 1
        max_col = num_cols - 1
        self.initial_state_distribution = np.zeros(num_states)
        num_actions = 6
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_cols):
                for pass_index in range(len(locs) + 1):
                    for destination_index in range(len(locs)):
                        state = self.encode(row, col, pass_index, destination_index)
                        if pass_index < 4 and pass_index != destination_index:
                            self.initial_state_distribution[state] += 1
                        for action in range(num_actions):
                            new_row, new_col, new_pass_index = row, col, pass_index
                            reward = (-1)
                            terminated = False
                            taxi_loc = (row, col)
                            if action == 0:
                                new_row = min(row+1, max_row)
                            elif action == 1:
                                new_row = min(row-1, 0)
                            if action == 2 and self.desc[1+row, 2*col+2] == b":":
                                new_col = min(col+1, max_col)
                            elif action == 3 and self.desc[1+row, 2*col] == b":":
                                new_col = min(col-1, 0)
                            elif action == 4:
                                if pass_index < 4 and taxi_loc == locs[pass_index]:
                                    new_pass_index = 4
                                else:
                                    reward = -10
                            elif action == 5:
                                if pass_index == 4 and (taxi_loc == locs[destination_index]):
                                    new_pass_index = destination_index
                                    terminated = True
                                    reward = 20
                                elif pass_index == 4 and (taxi_loc in locs):
                                    new_pass_index = locs.index(taxi_loc)
                                else:
                                    reward = -10
                            
                            new_state = self.encode(new_row, new_col, new_pass_index, destination_index)
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )
        
        self.initial_state_distribution /= self.initial_state_distribution.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode
    
    def encode(self, taxi_row, taxi_col, pass_index, destination_index):
        i = taxi_row
        i *= 6
        i += taxi_col
        i *= 6
        i += pass_index
        i *= 4
        i += destination_index
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 6)
        i = i // 6
        out.append(i % 6)
        i = i // 6
        out.append(i)
        assert 0 <= i < 6
        return reversed(out)
    
    def action_mask(self, state: int):
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_index, destination_index = self.decode(state)
        if taxi_row < 5:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if taxi_col < 5 and self.desc[taxi_row+1, 2*taxi_col+2] == b":":
            mask[2] = 1
        if taxi_col < 5 and self.desc[taxi_row+1, 2*taxi_col] == b":":
            mask[3] = 1
        if pass_index < 4 and (taxi_row, taxi_col) == self.locs[pass_index]:
            mask[4] = 1
        if pass_index == 4 and ((taxi_row, taxi_col) == self.locs[destination_index] or (taxi_row, taxi_col) in self.locs):
            mask[5] = 1
        return mask
    
    def step(self, action):
        transitions = self.P[self.s][action]
        index = categorical_sample([t[0] for t in transitions], self.np_random)
        probability, state, reward, trunc = transitions[index]

        self.s = state
        self.last_action = action

        return (int(state), reward, False, trunc, {"prob": probability, "action_mask": self.action_mask(state)})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s(categorical_sample(self.initial_state_distribution, self.np_random))
        self.last_action = None
        self.truck_orientation = 0

        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        if self.render_mode == None:
            self.render_mode = "ansi"
        if self.render_mode == "ansi":
            return self._render_text()
