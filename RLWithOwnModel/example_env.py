from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

MAP = [
    "+-----------+",
    "|R: | : : |G|",
    "| : | : : | |",
    "| : :W: : | |",
    "| | : | | : |",
    "| | : | | : |",
    "|Y| | : :B| |",
    "+-----------+",
]

class ExampleEnv(Env):
    metadata = {
        "render_modes": ["ansi", "rgb_array"],
        "render_fps": 4 
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 5), (2, 2), (5, 0), (5, 4)]

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
        for row in range(num_rows):
            for col in range(num_cols):
                for pass_index in range(len(locs) + 1):
                    for destination_index in range(len(locs)):
                        state = self.encode(row, col, pass_index, destination_index)
                        if pass_index < 5 and pass_index != destination_index:
                            self.initial_state_distribution[state] += 1
                        for action in range(num_actions):
                            new_row, new_col, new_pass_index = row, col, pass_index
                            reward = (-1)
                            terminated = False
                            taxi_loc = (row, col)
                            if action == 0:
                                new_row = min(row+1, max_row)
                            elif action == 1:
                                new_row = max(row-1, 0)
                            if action == 2 and self.desc[1+row, 2*col+2] == b":":
                                new_col = min(col+1, max_col)
                            elif action == 3 and self.desc[1+row, 2*col] == b":":
                                new_col = max(col-1, 0)
                            elif action == 4:
                                if pass_index < 5 and taxi_loc == locs[pass_index]:
                                    new_pass_index = 5
                                else:
                                    reward = -10
                            elif action == 5:
                                if pass_index == 5 and (taxi_loc == locs[destination_index]):
                                    new_pass_index = destination_index
                                    terminated = True
                                    reward = 20
                                elif pass_index == 5 and (taxi_loc in locs):
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
        i *= 5
        i += destination_index
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 5
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
        taxi_row, taxi_col, pass_index, destination_index = self.decode(self.state)
        
        def underline(x):
            return "_" if x == " " else x
        
        if pass_index < 5:
            out[1+taxi_row][2*taxi_col+1] = utils.colorize(
                out[1+taxi_row][2*taxi_col+1], "yellow", highlight=True
            )
            pos_i, pos_j = self.locs[pass_index]
            out[1+pos_i][2*pos_j+1] = utils.colorize(
                out[1+pos_i][2*pos_j+1], "blue", bold=True
            )
        else:
            out[1+taxi_row][2*taxi_col+1] = utils.colorize(
                underline(out[1+taxi_row][2*taxi_col+1]), "green", highlight=True
            )
        dest_i, dest_j = self.locs[destination_index]
        out[1+dest_i][2*dest_j+1] = utils.colorize(out[1+dest_i][2*dest_j+1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.last_action is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.last_action]}\n"
            )
        else:
            outfile.write("\n")
        
        with closing(outfile):
            return outfile.getvalue()
