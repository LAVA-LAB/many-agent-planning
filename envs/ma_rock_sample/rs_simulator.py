from __future__ import annotations

import copy
import math
import itertools
import random
from envs.ma_rock_sample.ma_rock_sample import Actions, MARockSample, RSState
from envs.simulator import Simulator
from envs.factored_simulator_helper import FactoredSimulators

import numpy as np

import scipy.stats.qmc as qmc

from graphs.cg import CoordinationGraph

class RockSampleSimulator(Simulator):
    
    def __init__(self, simulator : MARockSample, num_factors : int, graph : CoordinationGraph,  num_agents_per_factor = 2) -> None:
        super().__init__(simulator)
        self.n_agents = simulator.num_agents
        self._action_size = 4 + 1 + simulator.num_rocks
        self.true_initial_state = copy.deepcopy(self._simulator.init_state)
        self.sampler = qmc.LatinHypercube(d=2)# + self.n_agents * 2)
        self._num_factors = num_factors
        self._num_agents_per_factor = num_agents_per_factor
        self._graph = graph
        assert isinstance(self._action_size, int), f"Action size ({self._action_size}) should be an integer!"

    def check_state(self, s):
        if not isinstance(s, RSState):
            s = RSState(tuple([tuple(pos) for pos in np.array_split(s[:self.n_agents*2], self.n_agents)]), s[self.n_agents*2:])
        return s

    def generative_transition(self, s, a):
        """ 
        s', o, r ~ G(s, a)
        """
        a = np.array(a, dtype=int).ravel()

        new_state, o, r, done = self._simulator.step(self.check_state(s), a)

        # Return s', o, r, n_steps
        return new_state, [tuple(o)], r, not done

    def sample_initial(self, simple_belief=True):
        s = MARockSample.generate_instance(self._simulator.grid_size, self._simulator.num_rocks, self.n_agents)[0]
        if simple_belief:
            s.position = tuple(self.true_initial_state.position)
        return s

    def reward_model(self, state, act, next_state):
        # print(state, act)
        new_state, _, r, _ = self._simulator.step(self.check_state(state), np.array(act, dtype=int).ravel())
        # assert next_state == new_state, (next_state, new_state) # FIXME
        return r

    def is_final(self, state : RSState):
        return all([state.position[i][0] == self._simulator.grid_size for i in range(self.n_agents)])
            

    def obs_prob(self, state, act, obs):
        if isinstance(self, FactoredRockSampleSimulator):
            assert np.ndim(obs) > 1, obs
            obs = FactoredSimulators.factored_to_total(self._graph, obs)
        else:
            obs = np.array(obs).ravel()
            # [obs] = obs
        return self._simulator.obs_prob(state, np.array(act).ravel(), obs)

    def get_state(self):
        return self.true_initial_state

    def get_heuristic_action(self, state) -> list[int]:
        a = [None] * self.n_agents
        for i in range(self.n_agents):
            if state.position[i] in self._simulator.rock_locs and state.rocktypes[state.position[i]] > 0:
                a[i] = Actions.E_SAMPLE.value
            else:
                a[i] = Actions.E_EAST.value
        return a

    def get_random_joint_action(self) -> list[int]:
        return [random.randint(0, self._action_size-1) for _ in range(self.n_agents)]

    def get_random_action(self, state=None, heuristic=True) -> list[int]:
        if state is None or not heuristic:
            return self.get_random_joint_action()
        else:
            return self.get_heuristic_action(state)
    
    def action_size(self, agent : int) -> int:
        return self._action_size

    def get_possible_actions(self, factored=True):
        return itertools.product(range(self._action_size), repeat=self._num_agents_per_factor if factored else self.n_agents)

    def reset(self):
        self._simulator.reset()
    
    def get_state_indices(self, agent_id : int) -> set[int]:
        rock_ids = np.array_split(range(self.n_agents * 2, self.n_agents * 2 + self._simulator.num_rocks), self.n_agents)[agent_id]
        return {2*agent_id, (2*agent_id)+1}.union(rock_ids)

    def get_state_vector_size(self) -> int:
        return self.n_agents * 2 + self._simulator.num_rocks

class FactoredRockSampleSimulator(RockSampleSimulator):

    def __init__(self, simulator: MARockSample, num_factors: int, graph: CoordinationGraph, num_agents_per_factor=2) -> None:
        super().__init__(simulator, num_factors, graph, num_agents_per_factor)

    def generative_transition(self, s : tuple[int], a : list[int]) -> tuple[tuple[int], list[tuple], float, int]:
        """ 
        s', o, r ~ G(s, a)
        """
        # Create 1D array if nested/factored.
        new_state, [obs], reward, step, *_ = super().generative_transition(s, a)
        factored_obs = FactoredSimulators.total_to_factored(self._graph, obs)
        # Return s', (factored) o, r, n_steps
        return new_state, factored_obs, reward, step

    def factored_obs_prob(self, state, act, obs, ids):
        probs = 1
        for o, i in zip(obs, ids):
            probs *= self._simulator.individual_obs_prob(state, i, act[i], o, act)
        return probs

def build_mars(num_agents : int, horizon : int, joint : bool, size=7, num_rocks=8, num_factors : int = None, topology : str = 'teams', **kwargs) -> tuple[Simulator, CoordinationGraph]:
    sim = MARockSample(size, num_rocks, num_agents)
    agent_ids = list(range(num_agents))
    if topology in 'teams':
        assert num_agents % 2 == 0
        edges = [[i, i+1] for i in range(0,num_agents, 2) if i+1 in agent_ids]
    elif topology in 'line':
        edges = [[i, i+1] for i in agent_ids if i+1 in agent_ids]
    elif topology in 'dense':
        edges = FactoredSimulators.generate_random_dense_graph(num_agents).edges
    graph = CoordinationGraph(list(agent_ids), edges)
    if joint:
        sim = RockSampleSimulator(sim, len(edges), graph)
    else:
        sim = FactoredRockSampleSimulator(sim, len(edges), graph)
    return sim, graph
