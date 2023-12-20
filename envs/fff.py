from __future__ import annotations

from typing import Optional, Tuple, Union

import itertools

import numpy as np
import random
from envs.factored_simulator_helper import FactoredSimulators
from envs.simulator import Simulator
from graphs.cg import CoordinationGraph

import gym
from gym import spaces

from scipy.stats import qmc

import copy

class GeneralisedFireFighting(gym.Env):
    """
    Implemented as described in [1] F. A. Oliehoek, M. T. J. Spaan, and S. Whiteson, “Exploiting Locality of Interaction in Factored Dec-POMDPs,” p. 8.
    """

    def __init__(self, N, Nh, max_episode_length = None, n_fire_levels = 3) -> None: 
        self._num_agents = N
        self._num_houses = Nh
        self._nf = n_fire_levels  # In [1], the problem is defined with 3 fire levels.
        self._steps = 0
        self.fff = True # This is Generalised Firefighting, not standard firefighting.
        self._max_episode_length = max_episode_length
        self._initial_obs, _ = self.reset()
        self.sampler = qmc.LatinHypercube(d=self._num_houses)
        assert self._num_agents == self._num_houses - 1, "Generalised firefighting as in JAIR paper, with one less agents than houses. Can't guarantee correctness of the problem for other configurations."

        # Agent can stay at house or go to house n+1
        self._num_possible_actions_per_agent = 2
        self.action_space = spaces.Discrete(self._num_possible_actions_per_agent ** self._num_agents)

        self.observation_space = spaces.Discrete(2 ** self._num_agents)

        self.min_reward = -(self._num_houses * self._nf)
        self.max_reward = 0

    def reward_model(self, state) -> float:
        return -sum(state)

    def reward(self) -> float:
        """
        The (non-factored) reward of generalised firefighting is the negative sum of the fire levels of the houses.
        """
        return -sum(self._state)
        # return (self._num_houses * self._nf) - np.sum(self._state)

    def take_action(self, actions : np.ndarray) -> tuple[np.ndarray, float]:
        """
        Actions are the moves of every agent, which can be added to their positions modulo the number of houses to model the cyclic nature.

        Returns: observations as array and scalar reward.
        """
        assert np.sum(actions) <= self._num_agents, f"|A| = |N| and actions are in [0, 1] so the sum can't exceed the number of agents. Actions: {actions}, n_agents: {self._num_agents}"
        assert len(actions) == self._num_agents == len(self._agent_pos), (actions, self._num_agents, self._agent_pos)
        assert (self._agent_pos + np.array(actions, dtype=int)).max() <= (self._num_houses-1)
        self._agent_pos = (self._agent_pos + np.array(actions, dtype=int))
        self.update_state()
        obs = self.observe()
        self._reset_agent_pos()
        assert np.size(obs) == np.size(actions)
        return obs, self.reward()
    
    def _reset_agent_pos(self) -> None:
        """
        In Factored Firefighting, the agents are always assumed to be at 'their' house 
        before an action is taken to decide if they go to their rightmost house.

        In standard firefighting this is not always the case.
        """
        self._agent_pos = np.arange(0, self._num_agents, dtype=int)

    def update_state(self) -> None:
        """
        Updating state involves enumerating the houses and both increasing and decreasing their fire levels w.r.t. the number of fighters and neighbouring burning houses. 
        """
        current_state = self._state
        new_state = [None] * self._num_houses
        total_agents = 0
        for h in range(self._num_houses):
            agents_at_house = len(self._agent_pos[self._agent_pos == h])
            total_agents += agents_at_house

            curLevel = current_state[h]
            sameLevel = curLevel
            higherLevel = min(curLevel + 1, self._nf-1)
            lowerLevel  =  0 if (curLevel==0) else (curLevel-1)
            neighborIsBurning = self.neighbours_burning(h)

            probs = [None] * self._nf

            for nextLevel in range(self._nf):
                p2 = 0
                if agents_at_house == 0:
                    if neighborIsBurning:
                        if(nextLevel == sameLevel):
                            p2+=0.2
                        if(nextLevel == higherLevel):
                            p2+=0.8
                    elif (curLevel == 0): # //fire won't get ignited
                        if(0 == nextLevel):
                            p2=1.0
                        else: # not possible so we can quit...
                            p2=0.0
                    else: # normal burning house
                        if(nextLevel == sameLevel):
                            p2+=0.6
                        if(nextLevel == higherLevel):
                            p2+=0.4
                elif agents_at_house == 1:
                    if(neighborIsBurning):
                        if(nextLevel == sameLevel):
                            p2+=0.4
                        if(nextLevel == lowerLevel):
                            p2+=0.6 # // .6 prob of extuinguishing 1 fl
                    elif (curLevel == 0): # //fire won't get ignited
                        if(0 == nextLevel):
                            p2=1.0
                        else: # //not possible so we can quit...
                            p2=0.0
                    else: # //normal burning house
                        if(nextLevel == sameLevel):
                            p2+=0.0
                        if(nextLevel == lowerLevel):
                            p2+=1.0
                elif agents_at_house == 2:
                    # more than 1 agent: fire is extinguished
                    if(0 == nextLevel):
                        p2=1.0
                    else: # not possible so we can quit...
                        p2=0.0
                else:
                    raise ValueError("?")
                
                probs[nextLevel] = p2
            
            new_state[h] = random.choices(range(self._nf), weights=probs, k=1)[0]

        assert total_agents == self._num_agents
        self.state = new_state

    def observe(self) -> np.ndarray:
        """
        Agents only observe whether the house they are at is burning or not with some probability depending on the local environment.
        """
        obs = np.zeros(self._num_agents, dtype=int)
        for agent in range(self._num_agents):
            agent_h = self._agent_pos[agent]
            if self.is_burning(agent_h):
                if self._state[agent_h] > 1:
                    prob = 0.8
                else:
                    prob = 0.5
            else:
                prob = 0.2
            if random.random() < prob:
                obs[agent] = 1
        return obs

    def get_prob(self, firelevel, o):
        if firelevel == 0:
            prob = 0.2
        elif firelevel == 1:
            prob = 0.5
        elif firelevel == 2:
            prob = 0.8
        else:
            raise ValueError(f"{firelevel} + {o} = ?")

        if o == 0: # NO FLAMES
            return 1 - prob
        elif o == 1: # FLAMES
            return prob
        else:
            raise ValueError(f"{firelevel} + {o} = ?")

    def individual_obs_prob(self, state, agent_i, agent_a, agent_o, joint_a):
        """
        Individual observation prob of a single agent, given the state, agent idx, agent action, and agent observation.
        """
        firelevel = state[agent_i + agent_a]
        try:
            return self.get_prob(firelevel, agent_o)
        except ValueError as ve:
            print(state, firelevel, agent_i, agent_a, agent_o)
            raise ve
    
    def get_agent_locs(self, act):
        return np.arange(self._num_agents) + np.array(act, dtype=int)

    def obs_prob(self, state, act, obs, vector=False):
        """
        Joint observation probability given the state, joint actions and joint observations. 
        Can be returned as a vector of individual probabilities or a scalar by taking the product of the vector.
        """
        agent_loc = self.get_agent_locs(act)
        probs = np.zeros(self._num_agents)
        for a_i, (i_loc, o) in enumerate(zip(agent_loc, obs)):
            flames = state[i_loc]
            try:
                probs[a_i] = self.get_prob(flames, o)
            except ValueError as ve:
                print("Agent:", a_i, "Location:", i_loc, "Observation:", o, "Local state:", flames)
                raise ve
        return probs if vector else probs.prod()

    def sample_multiple_initial(self, n : int):
        """
        Stratified initial belief sampling.
        """
        samples = self.sampler.random(n=n)
        samples = qmc.scale(samples, np.zeros(self.sampler.d), np.ones(self.sampler.d) * self._nf).astype(int)
        return samples

    def decrease_fl(self, state : np.ndarray, h : int) -> None:
        """
        Decrease fire level if not zero yet.
        """
        if state[h] > 0:
            state[h] -= 1

    def increase_fl(self, state : np.ndarray, h : int) -> None:
        """
        Increase fire level if not at maximum level yet.
        """
        if state[h] < (self._nf-1):
            state[h] += 1

    def neighbours_burning(self, h : int) -> bool:
        """
        Return true if any of the neighbouring houses are burning (i.e. fire levels greater than 0).
        """
        if self.fff:
            # Generalised firefighting, need to check the borders.
            if h == 0:
                # h is the leftmost house
                return self._state[(h+1)] > 0
            elif h == (self._num_houses - 1):
                # h is the rightmost house
                return self._state[(h-1)] > 0
            else:
                return (self._state[(h+1)] + self._state[(h-1)]) > 0
        else:
            # If h is not a border house we can use the normal check.
            return self._state[(h+1)] + self._state[(h-1)] > 0

    def is_burning(self, h : int) -> bool:
        return self._state[h] > 0

    # GYM METHODS

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward = self.take_action(action)

        self._steps += 1

        truncated = False
        if self._max_episode_length is not None and self._steps == self._max_episode_length:
            truncated = True

        # If the reward is zero then no houses are burning.
        done = reward == 0

        return obs, reward, done, truncated, {"n_steps" : 1}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self._state = self.sample_initial_state()
        self._reset_agent_pos()
        return self.observe(), {}
    
    def sample_initial_state(self):
        """
        Initial state is intialised with uniform probability of the firelevels per house.
        """
        return np.random.randint(0, self._nf, size=self._num_houses, dtype=int)
    
    def render(self) -> None:
        raise NotImplementedError()

    # HELPERS

    @staticmethod
    def indices_to_joint(input : np.ndarray) -> int:
        """
        Return integer representation of binary actions, e.g. [1, 0] = 2 and [1, 1] = 3. 
        """
        assert np.sum(input) <= np.size(input), "Input should be binary array."
        return int("".join(str(x) for x in input), 2)

    def _joint_to_indices(self, input : int) -> np.ndarray:
        """
        See `joint_to_indices`.
        """
        return self.joint_to_indices(input, action_count=self.action_space.n)

    @staticmethod
    def joint_to_indices(input : int, action_count : int) -> np.ndarray:
        """
        Return binary representation of integer (joint actions), e.g. 2 = [1, 0] and 3 = [1, 1].

        It's a bit more complicated since the length of the array should be identical always. E.g. if the number of agents is 3, then all possible integers should return the same length binary array.
        """
        assert isinstance(input, int) and isinstance(action_count, int), (input, action_count)
        bit_width = len(format(action_count - 1, 'b'))
        return np.array([
            int(x) for x in bin(input)[2:].rjust(bit_width, '0')
            ])

    # GETTERS AND SETTERS

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, state) -> None:
        assert np.size(self._state) == np.size(state), f"New state should be of equal length. Original: {self._state}. Setter value: {state}."
        self._state = state

class JointFireFightingSimulator(Simulator):
    def __init__(self, simulator : GeneralisedFireFighting, precompute_all_states=False, num_actions_per_factor : int = 2) -> None:
        super().__init__(simulator)
        self.precompute_all_states = precompute_all_states
        if self.precompute_all_states:
            self.possible_states : list[np.ndarray] = list(itertools.product(range(self._simulator._nf), repeat=self._simulator._num_houses))
            assert np.shape(self.possible_states)[1] == self._simulator._num_houses
        self._action_size = 2
        self._num_actions_per_factor = num_actions_per_factor
    
    def reward_model(self, state, action, next_state):
        return self._simulator.reward_model(next_state)

    def obs_prob(self, state, act, obs):
        if isinstance(self, FactoredFireFightingSimulator):
            assert np.ndim(obs) > 1, obs
            obs = FactoredSimulators.factored_to_total(self._graph, obs)
        else:
            # act = self._simulator._joint_to_indices(np.array(act).item())
            act = np.array(act).ravel()
            # obs = self._simulator._joint_to_indices(np.array(obs).item())
            obs = np.array(obs).ravel()
        return self._simulator.obs_prob(state, act, obs)

    def get_random_joint_action(self) -> list[int]:
        return random.randint(0, self.get_num_joint_actions()-1)

    def get_random_action(self, state) -> list[int]:
        return self.get_random_joint_action()
    
    def check_state(self, s : Union[int, np.integer, tuple[int]]) -> list[int]:
        if isinstance(s, (int, np.integer)):
            assert self.precompute_all_states, "If we didnt precompute all states and pass an integer as state, then we can only return a random state. Note: this error should not occur when using particle filter(s)."
            # It's an integer representation of the state a.k.a. an index of possible states.
            return list(self.possible_states[s]) if self.precompute_all_states else np.random.randint(0, self._simulator._nf, size=self._simulator._num_houses)
        return list(s)

    def generative_transition(self, s, a, true_step=False):
        """ 
        s', o, r ~ G(s, a)
        """
        try:
            # Numpy's _.item() checks if the array is of length 1.
            a = np.array(a).item()
            # self._simulator.state = s
            ja = self._simulator._joint_to_indices(a)
        except ValueError:
            # Factored Statistics, the action is already a vector of individual actions.
            ja = np.array(a).ravel()
            assert len(ja) == self._simulator._num_agents, (ja, a)
        # Set state
        self._simulator.state = self.check_state(s)
        obs, reward = self._simulator.take_action(ja)
        steps = 0 if reward == 0 else 1
        # if not true_step:
        #     reward = (reward - self._simulator.min_reward) / (self._simulator.max_reward - self._simulator.min_reward)
        # obs = GeneralisedFireFighting.indices_to_joint(obs)
        new_state = np.array(self._simulator.state, dtype=int)
        # Return s', o, r, n_steps
        return tuple(new_state), [tuple(obs)], reward, steps

    def action_size(self, agent : int) -> int:
        return self._action_size

    def get_possible_actions(self, factored=True):
        if factored:
            return itertools.product([0, 1], repeat=self._num_actions_per_factor)
        else:
            return itertools.product([0, 1], repeat=self._simulator._num_agents)
            return range(self.get_num_joint_actions())

    def get_num_states(self) -> int:
        return self._simulator._nf ** self._simulator._num_houses

    def get_num_joint_actions(self) -> int:
        return self._simulator.action_space.n

    def get_num_joint_observations(self) -> int:
        return self._simulator.observation_space.n

    def initial_belief_at(self, s) -> float:
        return 1 / self.get_num_states()

    def get_dummy_initial_belief_vector(self) -> np.ndarray:
        return np.ones(self.get_num_states()) / self.get_num_states()

    def sample_initial(self):
        state = self._simulator.sample_initial_state()
        while self.is_final(state):
            state = self._simulator.sample_initial_state()
        return tuple(state)

    def get_state_indices(self, agent_id : int) -> set[int]:
        return {agent_id, agent_id+1}

    def get_state_vector_size(self) -> int:
        return self._simulator._num_houses

    def is_final(self, state):
        return sum(state) == 0

class FactoredFireFightingSimulator(JointFireFightingSimulator):
    def __init__(self, simulator : GeneralisedFireFighting, num_factors : int, graph : CoordinationGraph, num_actions_per_factor : int = 2) -> None:
        super().__init__(simulator, num_actions_per_factor=num_actions_per_factor)
        self._simulator = simulator
        self._num_factors = num_factors
        self._graph = graph
        assert self._num_factors == self._graph.num_edges
    
    def get_random_action(self, state) -> list[int]:
        return [random.randint(0, self._action_size-1) for _ in range(self._graph.num_agents)]
        # return self._simulator._joint_to_indices(self.get_random_joint_action()) # also works
    
    def factored_obs_prob(self, state, act, obs, ids):
        probs = 1
        for o, i in zip(obs, ids):
            probs *= self._simulator.individual_obs_prob(state, i, act[i], o, act)
        return probs
    
    def generative_transition(self, s : tuple[int], a : list[int], true_step=False) -> tuple[tuple[int], list[tuple], float, int]:
        """ 
        s', o, r ~ G(s, a)
        """
        # Set the state for Monte Carlo based simulation:
        self._simulator.state = list(super().check_state(s))

        # Create 1D array if nested/factored.
        obs, reward = self._simulator.take_action(np.array(a).ravel())
        steps = 0 if reward == 0 else 1
        # if not true_step:
        #     reward = (reward - self._simulator.min_reward) / (self._simulator.max_reward - self._simulator.min_reward)

        factored_obs = FactoredSimulators.total_to_factored(self._graph, obs)
        # factored_obs = [tuple(obs) for _ in range(self._num_factors)]

        # Return s', (factored) o, r, n_steps
        return tuple(self._simulator.state), factored_obs, reward, steps

########################### UTILS ########################################

def build_fff(num_agents : int, horizon : int, joint : bool = False, fire_levels = 3, ring=True, **kwargs) -> tuple[Simulator, CoordinationGraph]:
    fff = GeneralisedFireFighting(num_agents, num_agents+1, n_fire_levels=fire_levels)

    agent_ids = list(range(num_agents))

    edges = [(i, (i+1) % num_agents) for i in agent_ids] if ring else [[i, i+1] for i in agent_ids if i+1 in agent_ids]
    cg = CoordinationGraph(agent_ids, edges)

    # cg = FactoredSimulators.generate_random_dense_graph(num_agents)
    # edges = cg.edges

    if joint: 
        return JointFireFightingSimulator(fff), cg

    fsim = FactoredFireFightingSimulator(fff, len(edges), cg, 2)

    return fsim, cg
