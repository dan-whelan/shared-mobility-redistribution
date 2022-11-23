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
        self.initial_state_distrib = np.zeros(num_states)
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
                            self.initial_state_distrib[state] += 1
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
                                if pass_index < 4

