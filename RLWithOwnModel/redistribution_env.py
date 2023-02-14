from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import random

import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

MAP = [
    "+-----------+",
    "|D: | : : |D|",
    "| : | : : | |",
    "| : :D: : | |",
    "| | : | | : |",
    "| | : | | : |",
    "|D| | : :D| |",
    "+-----------+",
]

class RedistributionEnv(Env):
    metadata = {
        "render_modes": ["ansi", "rgb_array"]
    }


    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")
        self.num_of_docks = 5

        self.down = 0
        self.up = 1
        self.right = 2
        self.left = 3
        self.pickup = 4
        self.dropoff = 5

        self.dock_position = 0
        self.bikes_at_loc = 1

        self.locs = locs = [[(0, 0), 0], [(0, 5), 0], [(2, 2), 5], [(5, 0), 5], [(5, 4), 5]]
        self.visited = [False for i in range(self.num_of_docks)]
        self.docks_remaining = self.num_of_docks
        self.balancing_fig = 3
        self.bikes_on_truck = 0

        self.num_states = num_states = 36
        self.num_rows = num_rows = 6
        self.num_cols = num_cols = 6
        self.state_distribution = [[0 for i in range(2)] for j in range(num_states)]
        num_actions = 6
        
        curr_dock = 0
        curr_index = 0
        for row in range(num_rows):
            for col in range(num_cols):
                state = self.encode(row, col)
                self.state_distribution[curr_index][0] = state
                for dock in range(len(locs)):
                    if locs[dock][self.dock_position] == (row, col):
                        self.state_distribution[curr_index][1] = curr_dock
                        curr_dock += 1
                        break
                    else:
                        self.state_distribution[curr_index][1] = None
                curr_index += 1     

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode
    
    def encode(self, taxi_row, taxi_col):
        i = taxi_row
        i *= self.num_cols
        i += taxi_col
        return i

    def decode(self, i):
        taxi_col = i % self.num_cols
        i = i // self.num_cols
        taxi_row = i % self.num_rows
        assert 0 <= i < self.num_rows
        return taxi_row, taxi_col, 
    
    def step(self, state, action):
        row, col = self.decode(state)
        new_row, new_col = row, col
        reward = -1
        terminated = False
        if action == self.down:
            new_row = min(row+1, self.num_rows-1)
        elif action == self.up:
            new_row = max(row-1, 0)
        elif action == self.right and self.desc[1+row, 2*col+2] == b":":
            new_col = min(col+1, self.num_cols-1)
        elif action == self.left and self.desc[1+row, 2*col] == b":":
            new_col = max(col-1, 0)
        elif self.state_distribution[state][1] != None:
            dock = self.state_distribution[state][1]
            if action == self.pickup and self.visited[dock] == False:
                reward = 1
                moving_bikes = self.locs[dock][self.bikes_at_loc] - self.balancing_fig
                if moving_bikes < 0:
                    reward = -10
                else:
                    self.locs[dock][self.bikes_at_loc] -= moving_bikes
                    self.docks_remaining -= 1
                    self.visited[dock] = True
            elif action == self.dropoff and self.visited[dock] == False:
                reward = 1
                moving_bikes = self.balancing_fig - self.locs[dock][self.bikes_at_loc]
                if moving_bikes < 0:
                    reward = -10
                else:
                    self.locs[dock][self.bikes_at_loc] += moving_bikes
                    self.docks_remaining -= 1
                    self.visited[dock] = True
        if self.docks_remaining == 0:
            reward = 20
            terminated = True
        new_state = self.encode(new_row, new_col)
        return new_state, reward, terminated

    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.state = random.randint(0, self.num_states-1)
        self.last_action = None
        self.locs = [[(0, 0), 0], [(0, 5), 0], [(2, 2), 5], [(4, 5), 5], [(5, 0), 5]]
        self.visited = [False for i in range(self.num_of_docks)]
        self.balancing_fig = 3
        self.docks_remaining = self.num_of_docks

        return int(self.state)

    def render(self):
        if self.render_mode == None:
            self.render_mode = "ansi"
        if self.render_mode == "ansi":
            return self._render_text()
    
    def _render_text(self):
        description = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in description]
        taxi_row, taxi_col = self.decode(self.state)
        
        out[1+taxi_row][2*taxi_col+1] = utils.colorize(
            out[1+taxi_row][2*taxi_col+1], "yellow", highlight=True
        )

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.last_action is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.last_action]}\n"
            )
        else:
            outfile.write("\n")
        
        with closing(outfile):
            return outfile.getvalue()