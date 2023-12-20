from __future__ import annotations
import copy

from enum import Enum
from functools import reduce
import math
import random
import numpy as np

from envs.simulator import Simulator

def getX(state : RSState, index : int):
    return state.position[index][0]

def getY(state : RSState, index : int):
    return state.position[index][1]

# class Compass(Enum):
#     EAST = 1
#     NORTH = 2
#     SOUTH = 3
#     WEST = 4

class Compass(Enum):
    EAST = 1
    NORTH = 0
    SOUTH = 2
    WEST = 3

class Actions(Enum):
    E_SAMPLE = 4
    E_SOUTH = 2
    E_EAST = 1
    E_WEST = 3
    E_NORTH = 0

class Obs(Enum):
    E_BAD = 0
    E_GOOD = 1
    E_NONE = 2

NORTH = np.array([0, 1])
SOUTH = np.array([0, -1])
WEST = np.array([-1, 0])
EAST = np.array([1, 0])
TRANSLATION = {
    Compass.EAST : EAST,
    Compass.WEST : WEST,
    Compass.NORTH : NORTH,
    Compass.SOUTH : SOUTH
}
# STAY = np.array([0, 0])

class RSState():
    """Sampled from https://github.com/h2r/pomdp-py/blob/master/pomdp_problems/rocksample/rocksample_problem.py"""

    def __init__(self, position, rocktypes):
        """
        position (tuple): (x,y) position of the rover on the grid.
        rocktypes: tuple of size k. Each is either Good or Bad.
        terminal (bool): The robot is at the terminal state.
        (It is so true that the agent's state doesn't need to involve the map!)
        x axis is horizontal. y axis is vertical.
        """
        self.position = tuple(position)
        if type(rocktypes) != tuple:
            rocktypes = tuple(rocktypes)
        self.rocktypes = tuple(rocktypes)

    def __hash__(self):
        return hash((self.position, self.rocktypes))
    def __eq__(self, other):
        if isinstance(other, RSState):
            return self.position == other.position\
                and self.rocktypes == other.rocktypes
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "RSState(%s | %s)" % (str(self.position), str(self.rocktypes))

    def to_vector(self, array=False) -> tuple[int] | np.ndarray:
        return np.concatenate([np.concatenate(self.position), self.rocktypes]) if array else self.position + self.rocktypes

class MARockSample:
    
    """
    Adopted from the Hyp-DESPOT implementation:
    https://github.com/AdaCompNUS/hyp-despot/blob/8d10262ff1a188d963969b180639d56032774a17/src/HyP_examples/ma_rock_sample/src/ma_rock_sample/ma_rock_sample.cpp
    """

    def __init__(self, n, k, num_agents, half_eff_dist = 20) -> None:
        self.num_agents = num_agents
        self.grid_size = n
        self.num_rocks = k
        self.half_efficiency_distance = half_eff_dist
        self.reset()
        self.MAX_COORD_BIT = 10
        self.COORD_BIT_MASK = self.ROB_TERMINAL_ID = ((1 << self.MAX_COORD_BIT)-1)
        self.MAX_ACTION_BIT = 5
        self.ACTION_BIT_MASK = ((1 << self.MAX_ACTION_BIT)-1)
        self.MAX_OBS_BIT = 3
        self.OBS_BIT_MASK = ((1 << self.MAX_OBS_BIT)-1)

    def reset(self):
        self.init_state, self.rock_locs = MARockSample.generate_instance(self.grid_size, self.num_rocks, self.num_agents)
        self.rock_loc_to_id = {v : k for (k, v) in self.rock_locs.items()}

    def setRobObs(self, obs : Obs, setter : int, index : int) -> int:
        return obs + ((Obs)(setter -self.getRobObs(obs, index)) << (index*self.MAX_OBS_BIT))
    
    def getRobObs(self, obs : Obs, rid : int) -> int:
        return (obs >>rid*self.MAX_OBS_BIT) & self.OBS_BIT_MASK

    def step(self, rockstate : RSState, action : list[int]):
        rand_num = random.random()
        reward = 0
        obs=[Obs.E_NONE.value] * self.num_agents
        #Update each of the robot
        new_positions = list(rockstate.position)
        new_rock_types = list(rockstate.rocktypes)
        new_is_terminal = [False] * self.num_agents
        for i in range(self.num_agents):
            _ix = getX(rockstate, i)
            if(not _ix == self.grid_size): # if not terminal
                agent_action = action[i]
                if (agent_action < Actions.E_SAMPLE.value): # Move
                    _iy = getY(rockstate, i)
                    if agent_action == Compass.EAST.value:
                        if (_ix + 1 < self.grid_size):
                            _ix += 1
                        else:
                            _ix += 1
                            reward += +10
                            new_is_terminal[i] = True

                    elif agent_action == Compass.NORTH.value:
                        if (_iy + 1 < self.grid_size):
                            _iy += 1
                        else:
                            reward += -100

                    elif agent_action == Compass.SOUTH.value:
                        if (_iy - 1 >= 0):
                            _iy -= 1
                        else:
                            reward += -100

                    elif agent_action == Compass.WEST.value:
                        if (_ix - 1 >= 0):
                            _ix -= 1
                        else:
                            reward += -100

                    else:
                        raise ValueError("Should not occur!")

                    # Update location
                    new_positions[i] = (_ix, _iy)

                if (agent_action == Actions.E_SAMPLE.value) : # Sample
                    rock : int = self.rock_loc_to_id.get(rockstate.position[i], -1)
                    if (rock >= 0) :
                        if (rockstate.rocktypes[rock] == 1):
                            reward += +10
                        elif (rockstate.rocktypes[rock] == 0):
                            reward += -10
                        else:
                            raise ValueError("Should not occur.")
                        # self.sampleRock(rockstate, rock)
                        new_rock_types[rock] = 0 # Sampled rock so set to bad.
                    else :
                        reward += -100

                if (agent_action > Actions.E_SAMPLE.value): #debugging
                    rob_obs : int = 0
                    rock : int = agent_action - Actions.E_SAMPLE.value - 1
                    distance : float = math.dist(rockstate.position[i], self.rock_locs[rock])
                    efficiency : float = (1 + pow(2, -distance / self.half_efficiency_distance)) * 0.5
                    if (rand_num < efficiency):
                        rob_obs = rockstate.rocktypes[rock] & Obs.E_GOOD.value
                    else:
                        rob_obs = not (rockstate.rocktypes[rock] & Obs.E_GOOD.value)
                    obs[i] = int(rob_obs)
            else:
                new_is_terminal[i] = True
        return RSState(new_positions, new_rock_types), obs, reward, all(new_is_terminal)

    def numActions(self) -> int:
        return math.pow(self.robNumAction(), self.num_agents)

    def individual_obs_prob(self, rockstate : RSState, i, agent_action, rob_obs, joint_a) -> float:
        prob = 1
        if not isinstance(rockstate, RSState):
            rockstate = RSState(tuple([tuple(pos) for pos in np.array_split(rockstate[:self.num_agents*2], self.num_agents)]), rockstate[self.num_agents*2:])
        if(not getX(rockstate, i) == self.grid_size): # if not terminal
            if (agent_action <= Actions.E_SAMPLE.value):
                prob *= (rob_obs == Obs.E_NONE.value)
            elif (rob_obs != Obs.E_GOOD.value and rob_obs != Obs.E_BAD.value):
                prob *= 0
            else:
                rock : int = agent_action - Actions.E_SAMPLE.value - 1
                distance : float = math.dist(rockstate.position[i], self.rock_locs[rock])
                efficiency : float = (1 + math.pow(2, -distance / self.half_efficiency_distance)) * 0.5
                prob *=  efficiency if ((rockstate.rocktypes[rock] & 1) == rob_obs) else (1 - efficiency)
        return prob        

    def obs_prob(self, rockstate, action : list[int], obs, vector=False) -> float:
        probs = [None] * self.num_agents
        #calculate prob for each robot, multiply them together
        for i in range(self.num_agents):
            agent_action : int = action[i]
            rob_obs : int = obs[i]
            probs[i] = self.individual_obs_prob(rockstate, i, agent_action, rob_obs, None)
        return probs if vector else reduce(lambda x, y: x * y, probs, 1)

    def render(self, rockstate : RSState) -> None:
        string = "\n______ID______\n"
        rover_position = rockstate.position
        rocktypes = rockstate.rocktypes
        # Rock id map
        print
        for y in range(self.grid_size):
            for x in range(self.grid_size+1):
                char = ".."
                if x == self.grid_size:
                    char = ">"
                if (x,y) in self.rock_locs.values():
                    n = (self.rock_loc_to_id[(x,y)])
                    if n < 10:
                        char = "0"+str(n)
                    else:
                        char = str(n)
                for i in range(self.num_agents):
                    if (x,y) == rover_position[i]:
                        char = f"R{i}"
                string += char
                string += " "
            string += "\n"
        string += "_____G/B_____\n"
        # Good/bad map
        for y in range(self.grid_size):
            for x in range(self.grid_size+1):
                char = ".."
                if x == self.grid_size:
                    char = ">"
                if (x,y) in self.rock_locs.values():
                    if rocktypes[self.rock_loc_to_id[(x,y)]] == 1:
                        char = "$$"
                    else:
                        char = "XX"
                for i in range(self.num_agents):
                    if (x,y) == rover_position[i]:
                        char = f"R{i}"
                string += char
                string += " "
            string += "\n"
        print(string)

    @staticmethod
    def random_free_location(n, not_free_locs):
        """returns a random (x,y) location in nxn grid that is free."""
        while True:
            loc = (random.randint(0, n-1),
                   random.randint(0, n-1))
            if loc not in not_free_locs:
                return loc

    def in_exit_area(self, pos):
        return pos[0] == self.grid_size

    @staticmethod
    def get_random_rover_positions(n : int, num_agents : int):
        """
        Random rover positions on the left-most side (0, y) with rejection sampling.
        """
        rover_positions = []
        while len(rover_positions) < num_agents:
            pos = (0, random.randint(0, n-1))
            if pos not in rover_positions:
                rover_positions.append(pos)
        return tuple(rover_positions)

    @staticmethod
    def generate_instance(n, k, num_agents=2):
        """Returns init_state and rock locations for an instance of RockSample(n,k)"""

        if n == k == 11:
            rocks = [(0, 3), (0, 7), (1, 8), (2, 4), (3, 3), (3, 8), (4, 3), (5, 8), (6, 1), (9, 3), (9, 9)]
            if num_agents == 2:
                rover_positions = [(0,6), (0,4)]
            else:
                rover_positions = MARockSample.get_random_rover_positions(n, num_agents)
            rock_locs = {i : loc for i, loc in enumerate(rocks)}
        elif n == k == 15:
            rocks = [(0, 4), (0, 8), (1, 10), (3, 5), (4, 4), (4, 10), (5, 3), (7, 10), (7, 1), (14, 5), (11, 12), (12, 2), (2, 6), (6, 14), (9, 11)]
            if num_agents == 2:
                rover_positions = [(0,8), (0,6)]
            else:
                rover_positions = MARockSample.get_random_rover_positions(n, num_agents)
            rock_locs = {i : loc for i, loc in enumerate(rocks)}
        else:
            rover_positions = MARockSample.get_random_rover_positions(n, num_agents)
            rock_locs = {}  # map from rock location to rock id
            for i in range(k):
                loc = MARockSample.random_free_location(n, set(rock_locs.keys()) | set({rover_positions}))
                rock_locs[i] = loc

        # random rocktypes
        rocktypes = tuple((random.randint(0, 1)) for _ in range(k))

        # Ground truth state
        init_state = RSState(rover_positions, rocktypes)

        return init_state, rock_locs
