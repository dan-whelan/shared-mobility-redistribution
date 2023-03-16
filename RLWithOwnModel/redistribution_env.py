from contextlib import closing
from io import StringIO
from os import path
from typing import Optional, Tuple

import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

MAP = [
    "+-----------+",
    "|D: | : : | |",
    "| : | : : | |",
    "| : :D: : | |",
    "| | : | | : |",
    "| | : | | : |",
    "| | | : :D| |",
    "+-----------+",
]

class RedistributionEnv(Env):
    metadata = {
        "render_modes": ["ansi", "rgb_array"]
    }


    def __init__(self, num_docks=3, balanced_bikes=4, num_rows=6, num_cols=6, num_actions = 6, render_mode="ansi", max_step_per_episode=200):
        self.num_docks = num_docks
        self.balanced_bikes = balanced_bikes
        self.num_dock_states = 3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.observation_space = spaces.Discrete((num_rows * num_cols) * ((num_docks + 1) * self.num_dock_states))
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

        self.locs = [(0, 0), (2, 2), (5, 4)]
        self.bike_state = [0, 4, 8]
        initial_state = self.to_state(3, 3, self._calc_docks_for_rebalancing(), )
        self.initial_state_distribution = np.zeros(self.observation_space.n)

        self.initial_state_distribution[initial_state] += 1.0

        self.render_mode = render_mode
    
    def to_state(self, truck_row, truck_col, docks_for_rebalancing = 0, dock_state = 0):
        i = truck_row
        i *= self.num_cols
        i += truck_col
        i *= self.num_dock_states
        i += dock_state
        i *= self.num_docks
        i += docks_for_rebalancing
        return i

    def from_state(self, i):
        _ = i % self.num_docks
        i = i // self.num_docks
        _ = i % self.num_dock_states
        i = i // self.num_dock_states
        truck_col = i % self.num_cols
        i = i // self.num_cols
        truck_row = i % self.num_rows
        assert 0 <= i < self.num_rows
        return truck_row, truck_col
    
    def step(self, action):
        self.lastaction = action
        self.timestep +=1

        dock_index = 0

        truck_loc = row, col = (self.curr_row, self.curr_col)
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
                if self.bike_state[dock_index] > self.balanced_bikes:
                    bikes_out = self.bike_state[dock_index] - (self.balanced_bikes)
                    self.bike_state[dock_index] -= bikes_out
                    reward = 10
                else:
                    reward = -2
            else:
                reward = -5
        elif action == self.dropoff:
            if truck_loc in self.locs:
                dock_index = self.locs.index(truck_loc)
                if self.bike_state[dock_index] < self.balanced_bikes:
                    bikes_out = (self.balanced_bikes) - self.bike_state[dock_index]
                    self.bike_state[dock_index] += bikes_out
                    reward = 10
                else:
                    reward = -2
            else:
                reward = -5
        
        if len(set(self.bike_state)) == 1:
            reward = 40
            done = True

        if self.timestep >= self.max_timestep:
            done = True

        if (new_row, new_col) in self.locs:
            self.state = self.to_state(new_row, new_col, True, self.locs.index((new_row, new_col)))
        else:
            self.state = self.to_state(new_row, new_col)
        
        self.curr_row, self.curr_col = new_row, new_col

        return self.state, reward, done

    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.state = categorical_sample(self.initial_state_distribution, self.np_random)
        self.lastaction = None
        self.timestep = 0
        self.bike_state = [0, 4, 8]
        self.curr_row = 3
        self.curr_col = 3

        return int(self.state)

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
    
    def _calc_docks_for_rebalancing(self):
        bikes_present = set(self.bike_state)
        if self.balanced_bikes in bikes_present:
            if {i > self.balanced_bikes for i in bikes_present}:
                return (len(bikes_present)-1), self.overbalanced
            return (len(bikes_present)-1)
        return len(bikes_present)

if __name__ == '__main__':
    env = RedistributionEnv()


class RedistributionEnv2(Env):

    def __init__(self, num_docks=3, balanced_bikes=4, num_rows=6, num_cols=6, num_actions = 6, render_mode="ansi", max_step_per_episode=200, max_bikes_on_truck=5):
        self.num_docks = num_docks
        self.balanced_bikes = balanced_bikes
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.observation_space = spaces.Dict({
            "truck_position": spaces.Discrete(num_rows * num_cols),
            "bike_states": spaces.MultiDiscrete([(balanced_bikes*2)+1]*num_docks),
            "bikes_on_truck": spaces.Discrete(max_bikes_on_truck+1)
        })
        self.action_space = spaces.Discrete(6)
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
        
        self.locs = [(0, 0), (2, 2), (5, 4)]
        self.bike_state = [0, 4, 8]
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
                    reward = -2
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
                    reward = -2
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
        self.bike_state = [0, 4, 8]
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
