from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

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

        self.locs = locs = [[(0, 0), 0], [(0, 5), 0], [(2, 2), 5], [(5, 4), 5], [(5, 0), 5]]
        self.balancing_fig = 3
        self.bikes_on_truck = 0

        num_states = 1080
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
        #Change logic of for loop
        # - link bikes present to docks 
        docks_balanced = 0
        pickup_reward = self.num_of_docks
        dropoff_reward = self.num_of_docks
        for row in range(num_rows):
            for col in range(num_cols):
                for dock_index in range(len(locs)):
                    state = self.encode(row, col, dock_index)
                    if dock_index < self.num_of_docks and docks_balanced != self.num_of_docks:
                        self.initial_state_distribution[state] += 1
                    for action in range(num_actions):
                        new_row, new_col, new_dock_index = row, col, dock_index
                        reward = (-1)
                        terminated = False
                        truck_loc = (row, col)
                        if action == self.down:
                            new_row = min(row+1, max_row)
                        elif action == self.up:
                            new_row = max(row-1, 0)
                        elif action == self.right and self.desc[1+row, 2*col+2] == b":":
                            new_col = min(col+1, max_col)
                        elif action == self.left and self.desc[1+row, 2*col] == b":":
                            new_col = max(col-1, 0)
                        elif action == self.pickup:
                            if truck_loc == locs[dock_index][self.dock_position] and locs[dock_index][self.bikes_at_loc] > self.balancing_fig:
                                moving_bikes = locs[dock_index][self.bikes_at_loc] - self.balancing_fig
                                self.bikes_on_truck += moving_bikes
                                locs[dock_index][self.bikes_at_loc] -= moving_bikes
                                reward = pickup_reward
                                pickup_reward -= 1
                            else:
                                reward = -10
                        elif action == self.dropoff:
                            if truck_loc == locs[dock_index][self.dock_position] and locs[dock_index][self.bikes_at_loc] < self.balancing_fig:
                                moving_bikes = self.balancing_fig - locs[dock_index][self.bikes_at_loc]
                                locs[dock_index][self.bikes_at_loc] += moving_bikes
                                self.bikes_on_truck -= moving_bikes
                                reward = dropoff_reward
                                dropoff_reward -= 1
                                docks_balanced = 0
                                for dock in range(self.num_of_docks):
                                    if locs[dock][self.bikes_at_loc] == self.balancing_fig:
                                        docks_balanced+=1
                                if docks_balanced == self.num_of_docks and self.bikes_on_truck == 0:
                                    terminated = True
                                    reward = 20   
                            else:
                                reward = -10

                        new_state = self.encode(new_row, new_col, new_dock_index)
                        self.P[state][action].append(
                            (1.0, new_state, reward, terminated)
                        )
    
        self.initial_state_distribution /= self.initial_state_distribution.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode
    
    def encode(self, taxi_row, taxi_col, dock_index):
        i = taxi_row
        i *= 6
        i += taxi_col
        i *= 6
        i += dock_index
        i *= self.num_of_docks
        return i

    def decode(self, i):
        out = []
        i = i // self.num_of_docks
        out.append(i % 6)
        i = i // 6
        out.append(i % 6)
        i = i // 6
        out.append(i)
        return reversed(out)
    
    def step(self, action):
        transitions = self.P[self.state][action]
        index = categorical_sample([t[0] for t in transitions], self.np_random)
        _, state, reward, trunc = transitions[index]

        self.state = state
        self.last_action = action

        done = False

        if reward == 20:
            done = True

        return (int(state), reward, done, trunc)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = categorical_sample(self.initial_state_distribution, self.np_random)
        self.last_action = None
        self.truck_orientation = 0
        self.locs = [[(0, 0), 0], [(0, 5), 0], [(2, 2), 5], [(4, 5), 5], [(5, 0), 5]]
        self.bikes_on_truck = 0
        self.balancing_fig = 3

        return int(self.state)

    def render(self):
        if self.render_mode == None:
            self.render_mode = "ansi"
        if self.render_mode == "ansi":
            return self._render_text()
    
    def _render_text(self):
        description = self.desc.copy().tolist()
        print(description)
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in description]
        taxi_row, taxi_col, dock_index = self.decode(self.state)
        
        def underline(x):
            return "_" if x == " " else x
        
        if dock_index < self.num_of_docks:
            out[1+taxi_row][2*taxi_col+1] = utils.colorize(
                out[1+taxi_row][2*taxi_col+1], "yellow", highlight=True
            )
        else:
            out[1+taxi_row][2*taxi_col+1] = utils.colorize(
                underline(out[1+taxi_row][2*taxi_col+1]), "green", highlight=True
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
