from __future__ import annotations
import random
from envs.factored_simulator_helper import FactoredSimulators

from envs.simulator import Simulator
from envs.capturetarget.capturetarget import CaptureTarget

from graphs.cg import CoordinationGraph

import math
import itertools

import numpy as np

import scipy.stats.qmc as qmc

import copy

class CaptureTargetSimulator(Simulator):
    def __init__(self, simulator : CaptureTarget, num_factors : int, graph : CoordinationGraph, num_agents_per_factor : int = 2) -> None:
        super().__init__(simulator)
        self._simulator : CaptureTarget = simulator # for ide code completion
        self.n_agents = simulator.n_agent
        self._action_size = simulator.action_space[0].n
        self.true_initial_state = copy.deepcopy(self._simulator.get_state())
        self.sampler = qmc.LatinHypercube(d=2)# + self.n_agents * 2)
        self._num_factors = num_factors
        self._num_agents_per_factor = num_agents_per_factor
        self._graph = graph
        assert isinstance(self._action_size, int), f"Action size ({self._action_size}) should be an integer!"
    
    def generative_transition(self, s, a):
        """ 
        s', o, r ~ G(s, a)
        """
        self._simulator.set_state(s)
        a = np.array(a, dtype=int).ravel()
        o, r, done, *_ = self._simulator.step(a)
        new_state = tuple(self._simulator.state)

        # Return s', o, r, n_steps
        return new_state, [tuple(o)], r, int(not done)

    def obs_prob(self, state, act, obs):
        # if isinstance(self, FactoredCaptureTargetSimulator):
            # assert np.ndim(obs) > 1, obs
            # obs = FactoredSimulators.factored_to_total(self._graph, obs)
        return self._simulator.obs_prob(state, np.array(act).ravel(), np.array(obs).ravel())

    @property
    def is_joint(self) -> bool:
        return True

    def get_state(self):
        return self._simulator.get_state()
    
    def set_state(self, state) -> None:
        self._simulator.set_state(state)

    def get_random_joint_action(self) -> list[int]:
        return [random.randint(0, self._action_size-1) for _ in range(self._simulator.n_agent)]

    def get_random_action(self, state=None, heuristic=True) -> list[int]:
        if state is None or not heuristic:
            return self.get_random_joint_action()
        else:
            return self.get_heuristic_action(state)

    def get_heuristic_action(self, state):
        prev_state = self._simulator.get_state()
        self._simulator.set_state(state)
        a = [None] * self.n_agents
        for a_id in range(self.n_agents):
            a[a_id] = self._simulator.get_heuristic_move(a_id).item()
        self._simulator.set_state(prev_state)
        return a

    def get_possible_actions(self, factored=True):
        return itertools.product(range(self._action_size), repeat=self._num_agents_per_factor if factored else self._simulator.n_agent)

    def transform(self, x):
        """Transform is used in Particle Reinvigoration, add noise to resampled particles to prevent collapse."""
        arr = np.array(x)
        a, t = arr[:-2], arr[-2:]
        return tuple(np.concatenate([a, (t + np.random.randint(0, self._simulator.y_len // 2, size=2)) % self._simulator.y_len]))

    def action_size(self, agent : int) -> int:
        return self._simulator.action_space[agent].n

    def get_state_indices(self, agent_id : int) -> set[int]:
        return {2*agent_id, (2*agent_id)+1, 2*self._simulator.n_agent, (2*self._simulator.n_agent)+1}

    def get_state_vector_size(self) -> int:
        return 2 * self._simulator.n_agent + 2

    def get_num_joint_observations(self) -> int:
        return sum(self._simulator.obs_size)

    def reward_model(self, state, action, next_state):
        return self._simulator.reward_of_state(next_state)

    def initial_belief_at(self, s) -> float:
        return 1 / self.get_num_states()

    def get_dummy_initial_belief_vector(self) -> np.ndarray:
        return np.array([1 / self.get_num_states() for _ in range(self.get_num_states())])
    
    def sample_multiple_initial(self, n : int):
        samples = self.sampler.random(n=n)
        samples = qmc.scale(samples, np.zeros(self.sampler.d), np.ones(self.sampler.d) * self._simulator.y_len).astype(int)
        if self.sampler.d > 2:
            return samples
        state = np.array([x for x in copy.deepcopy(self._simulator.get_state())[:-2]])
        states = np.tile(state, (n, *[1 for _ in range(state.ndim)]))
        return np.column_stack([states , samples])

    def sample_initial(self):
        return tuple([x for x in copy.deepcopy(self._simulator.get_state())[:-2]] + [random.randint(0, self._simulator.y_len), random.randint(0, self._simulator.y_len)])
        if True:
            return tuple(np.random.randint(0, self._simulator.y_len, size=(self._simulator.n_agent * 2) + 2).tolist())
        else:
            return tuple(self._simulator.get_state())

class FactoredCaptureTargetSimulator(CaptureTargetSimulator):
    def __init__(self, simulator : CaptureTarget, num_factors : int, graph : CoordinationGraph, num_agents_per_factor : int = 2) -> None:
        super().__init__(simulator, num_factors, graph, num_agents_per_factor)
    
    def generative_transition(self, s : tuple[int], a : list[int]) -> tuple[tuple[int], list[tuple], float, int]:
        """ 
        s', o, r ~ G(s, a)
        """
        # Create 1D array if nested/factored.
        new_state, [obs], reward, step, *_ = super().generative_transition(s, a)

        # Three agent two factor case:
        # obs = [a, b, c]
        # f_obs <- [[a,b], [b, c]]
        factored_obs = FactoredSimulators.total_to_factored(self._graph, obs)

        # Return s', (factored) o, r, n_steps
        return new_state, factored_obs, reward, step
    
    def is_final(self, state):
        self._simulator.set_state(state)
        return bool(self._simulator.reward())
        
    def factored_obs_prob(self, state, act, obs, ids):
        probs = 1
        for o, i in zip(obs, ids):
            probs *= self._simulator.individual_obs_prob(state, i, act[i], o, act)
        return probs

############ UTILS ##############

def list_of_tuple_to_set_via_numpy(x : list[tuple[int]]) -> set[int]:
    return set([i for sublist in np.array(x) for i in sublist])

def get_distance_pairs(ct_sim : CaptureTarget, num_agents : int):
    pos = ct_sim.agent_positions
    f = lambda x, y : (pos[x], pos[y])
    sorted_pairs = sorted(list(zip(itertools.combinations(range(num_agents), 2), itertools.starmap(math.dist, itertools.starmap(f, itertools.combinations(range(num_agents), 2))))), key=lambda x : x[1])
    return sorted_pairs
    

def build_distance_coordination_graph_teams(ct_sim : CaptureTarget, num_agents : int, num_factors : int = None) -> list[tuple[int]]:
    edges = []
    sorted_pairs = get_distance_pairs(ct_sim, num_agents)
    if num_factors is None:
        assert num_agents % 2 == 0
        num_factors = int(num_agents // 2)
    for edge, _ in sorted_pairs:
        taken = list_of_tuple_to_set_via_numpy(edges)
        if len(set(edge).intersection(taken)) == 0:
            edges.append(edge)
    assert len(edges) == num_factors, (edges, num_factors)
    return edges

def build_distance_coordination_graph_line(ct_sim : CaptureTarget, num_agents : int, num_factors : int = None) -> list[tuple[int]]:
    edges = []
    sorted_pairs = get_distance_pairs(ct_sim, num_agents)
    if num_factors is None:
        num_factors = num_agents-1
    print(sorted_pairs)
    for edge, _ in sorted_pairs:
        print(edge)
        edges.append(edge)
        if len(edges) == num_factors:
            break
    return edges

def rebuild_ct(ct : CaptureTarget, num_agents : int, horizon : int, joint : bool, num_factors : int = None, artificially_factored = False, fully_connected = False, teams = False, line = False):
    ct.terminate_step = horizon
    agent_ids = list(range(num_agents))
    edges = []

    if artificially_factored:
        assert num_agents == 4, "Only implemented for 4 agents as of yet."
        if num_factors is None:
            num_factors = 4
        # building capturetarget with artificial bounds, creating a coordination structure.
        for a_id in agent_ids:
            edges.append((a_id, (a_id + 1) % max(agent_ids)))
    elif fully_connected:
        edges = FactoredSimulators.generate_random_dense_graph(num_agents).edges
    elif teams:
        edges = build_distance_coordination_graph_teams(ct, num_agents)
    elif line:
        edges = [[i, i+1] for i in agent_ids if i+1 in agent_ids]
        # edges = build_distance_coordination_graph_line(ct, num_agents)
        print(edges)
    else:
        raise ValueError("?")

    graph = CoordinationGraph(agent_ids, sorted(edges))

    if joint:
        sim = CaptureTargetSimulator(ct, num_factors=len(edges), graph=graph)
    else:
        sim = FactoredCaptureTargetSimulator(ct, len(edges), graph, num_agents_per_factor=2)

    return sim, graph

def build_ct(num_agents : int, horizon : int, joint : bool, num_factors : int = None, topology : str = 'teams', **kwargs) -> tuple[Simulator, CoordinationGraph]:
    if 'team' in topology:
        artificially_factored = fully_connected = line = False
        teams = True
    elif topology in{'dense', 'fc', 'fullyconnected'}:
        artificially_factored = teams = line = False
        fully_connected = True
    elif 'factored' in topology:
        teams = fully_connected = line = False
        artificially_factored = True
    elif 'line' in topology:
        teams = fully_connected = artificially_factored = False
        line = True
    else:
        raise ValueError("The following topology is not recognised: " + topology)
    kwargs = {
        'artificially_factored' : artificially_factored,
        'fully_connected' : fully_connected,
        'teams' : teams,
        'line' : line
    }
    gd = (8, 8) if artificially_factored else (12, 12)
    ct_sim = CaptureTarget(n_target=1, n_agent=num_agents, terminate_step=horizon, restricted_movement=artificially_factored, target_flick_prob=0.3, grid_dim=gd)
    return rebuild_ct(ct_sim, num_agents, horizon, joint, num_factors = num_factors, **kwargs)
