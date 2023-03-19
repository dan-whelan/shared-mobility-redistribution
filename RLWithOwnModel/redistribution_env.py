from contextlib import closing
from io import StringIO
from os import path
from typing import Optional, Tuple

import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

MAP = [
    "+-------------------+",
    "|D: | : : |D: : : : |",
    "| : | : : | : | :D: |",
    "| : :D: : | : | : | |",
    "| | : | | : |D: : | |",
    "| | : | | : : | :D: |",
    "|D| | : :D| : : : : |",
    "+-------------------+",
]

class RedistributionEnv2(Env):

    def __init__(self, num_docks=8, max_bikes=8, num_rows=6, num_cols=6, num_actions = 6, render_mode="ansi", max_step_per_episode=200, max_bikes_on_truck=5):
        self.num_docks = num_docks
        self.max_bikes = max_bikes
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.observation_space = spaces.Dict({
            "truck_position": spaces.Discrete(num_rows * num_cols),
            "bike_states": spaces.MultiDiscrete([(max_bikes)+1]*num_docks),
            "bikes_on_truck": spaces.Discrete(max_bikes_on_truck+1)
        })
        self.action_space = spaces.Discrete(num_actions)
        self.desc = np.asarray(MAP, dtype="c")
        self.lastaction = None
        self.reward_range = (-5, 40)

        self.down = 0
        self.up = 1
        self.right = 2
        self.left = 3
        self.pickup = 4
        self.dropoff = 5

        self.timestep = 0
        self.balanced = 0
        self.underbalanced = 1
        self.overbalanced = 2

        self.max_timestep = max_step_per_episode
        self.max_bikes_on_truck = max_bikes_on_truck
        
        self.locs = [(0, 0), (0, 5), (1, 8), (2, 2), (3, 6), (4, 8), (5, 0), (5, 4)]
        self.bike_state = np.random.randint((self.max_bikes)+1, size=self.num_docks)
        count = 0
        for i in range(len(self.bike_state)):
            count += self.bike_state[i]
        self.balanced_bikes = count // self.num_docks

        self.bikes_on_truck = 0
        self.curr_row = 3
        self.curr_col = 3

    def step(self, action) -> Tuple[dict, float, bool]:
        self.lastaction = action
        self.timestep +=1

        dock_index = 0

        truck_loc = row, col = self._from_truck_position(self.state["truck_position"])
        done = False
        new_row, new_col = row, col
        reward = -1

        if action == self.down:
            new_row = min(row+1, self.num_rows-1)
        elif action == self.up:
            new_row = max(row-1, 0)
        elif action == self.left and self.desc[1+row, 2*col+2] == b":":
            new_col = min(col+1, self.num_cols-1)
        elif action == self.right and self.desc[1+row, 2*col] == b":":
            new_col = max(col-1, 0)
        elif action == self.pickup:
            if truck_loc in self.locs:
                dock_index = self.locs.index(truck_loc)
                if self.bike_state[dock_index] > self.balanced_bikes and self.bikes_on_truck < self.max_bikes_on_truck:
                    bikes_out = min(self.bike_state[dock_index] - self.balanced_bikes, self.max_bikes_on_truck - self.bikes_on_truck)
                    self.bike_state[dock_index] -= bikes_out
                    self.bikes_on_truck += bikes_out
                    reward = 10 - bikes_out
                else:
                    reward = -3
            else:
                reward = -5
        elif action == self.dropoff:
            if truck_loc in self.locs:
                dock_index = self.locs.index(truck_loc)
                if self.bike_state[dock_index] < self.balanced_bikes and self.bikes_on_truck > 0:
                    bikes_needed = self.balanced_bikes - self.bike_state[dock_index]
                    bikes_out = min(bikes_needed, self.bikes_on_truck)
                    self.bike_state[dock_index] += bikes_out
                    self.bikes_on_truck -= bikes_out
                    reward = 10 - bikes_out
                else:
                    reward = -3
            else:
                reward = -5
        
        if len(set(self.bike_state)) == 1:
            reward = 40
            done = True

        if self.timestep >= self.max_timestep:
            done = True
        
        self.curr_row, self.curr_col = new_row, new_col

        self.state = {
            "truck_position": self._to_truck_position(new_row, new_col),
            "bike_states": tuple(self.bike_state),
            "bikes_on_truck": self.bikes_on_truck
        }

        return self.state, reward, done
    
    def _to_truck_position(self, row, col):
        i = row
        i *= self.num_cols
        i += col
        return i 
    
    def _from_truck_position(self, i):
        row = i // self.num_cols
        col = i % self.num_cols
        return row, col

    def reset(self, *, seed: Optional[int] = None) -> dict:
        super().reset(seed=seed)
        self.lastaction = None
        self.timestep = 0
        self.bike_state = np.random.randint((self.max_bikes)+1, size=self.num_docks)
        count = 0
        for i in range(len(self.bike_state)):
            count += self.bike_state[i]
        self.balanced_bikes = count // self.num_docks
        self.bikes_on_truck = 0
        self.state = {
            "truck_position": self._to_truck_position(3, 3),
            "bike_states": self.bike_state,
            "bikes_on_truck": self.bikes_on_truck
        }
        return self.state

    def render(self):
        if self.render_mode == None:
            self.render_mode = "ansi"
        if self.render_mode == "ansi":
            return self._render_text()
    
    def _render_text(self):
        description = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line ]for line in description]
        taxi_row, taxi_col = self.curr_row, self.curr_col
        
        out[1+taxi_row][2*taxi_col+1] = utils.colorize(
            out[1+taxi_row][2*taxi_col+1], "yellow", highlight=True
        )

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")
        
        with closing(outfile):
            return outfile.getvalue()

if __name__ == '__main__':
    env = RedistributionEnv2()
    print(env.bike_state)
    