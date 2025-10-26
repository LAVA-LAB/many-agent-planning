from __future__ import annotations
from collections import defaultdict
import copy
from functools import partial, reduce
from multiprocessing import Pool, Process

import time
import math
import warnings
from typing import Optional, Union
import random

import numpy as np
import pandas as pd
from envs.capturetarget.capturetarget_simulator import CaptureTargetSimulator
from envs.factored_simulator_helper import FactoredSimulators
from mpomcp.action_selection import max_plus_ucb_final
from optimisation.jit_functions import UCB as opt_UCB
from mpomcp.definitions import Factor, FactoredStatistic, QNode, VNode, RootVNode
from mpomcp.particles import WPF, Particles, ParticleManager, WeightedParticles, ParticleFilterException
from envs.simulator import Simulator
from graphs.cg import CoordinationGraph
from graphs.et import EliminationTree

def multi_rollout(state, get_rollout_action, G, depth, max_depth, discount_factor):
    discount = 1
    total = 0
    nsteps = 1
    while depth < max_depth:
        a = get_rollout_action(None)
        next_state, _, reward, nsteps = G(state, a)
        total += reward * discount
        discount *= (discount_factor**nsteps)
        if nsteps == 0:
            break # terminal
        state = next_state
    return total

class ALPHA_POUCT:
    """
    Partially Observable Upper Confidence Trees (POUCT) with several extensions, i.e. support for:
        - Factored Trees + Factored Statistics,
        - Particles (as in POMCP), and,
        - Simulating weighted particles as in PFT
        - Progressive Widening on the observation space,
    """
    def __init__(
            self,
            num_iterations: int,
            simulator : Simulator,
            graph : CoordinationGraph = None,
            max_plus_iterations = 10,
            max_depth = 10,
            exploration_const = math.sqrt(2),
            num_factors=2,
            max_time=600,
            discount_factor=1.0,
            num_visits_init = 0,
            value_init = float(0),
            use_varel = False,
            use_particles = False,
            num_particles = 100,
            global_particles = False,
            dont_reuse_trees = False,
            fully_observable = False,
            progressive_widening = False,
            asymmetric_sampling = False,
            likelihood_sampling = False,
            weighted_particle_filtering = False,
            factored_statistics = False,
            factored_trees = True,
            surpress_warnings = False,
            spanning_tree = False,
            random_spanning_tree = False,
            factored_pf = False,
            pomcpow = False,
            pomcpow_global = False,
            pft = False,
            use_sim_particles = False,
            **kwargs,
    ) -> None:
        self._num_sims = num_iterations
        self._max_time = max_time # in seconds
        self.c = exploration_const
        self._max_depth = max_depth
        self._discount_factor =discount_factor
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._num_factors = num_factors
        self.simulator : Simulator = simulator
        self.FACTORED_STATISTICS = factored_statistics
        self.FACTORED_TREES = factored_trees
        self.FLAT_POMCP = not self.FACTORED_STATISTICS and not self.FACTORED_TREES

        if self.FACTORED_STATISTICS or self.FLAT_POMCP:
            # Factored Statistics, only actions are factored
            self.num_trees = 1
        elif self.FACTORED_TREES:
            # Factored Trees, tree for every factor
            self.num_trees = self._num_factors
        else:
            raise ValueError("One algorithm should be true.")

        self._factors : list[Factor] = [Factor(i, None, None) for i in range(self.num_trees)]

        self.dont_reuse_trees = dont_reuse_trees
        if self.dont_reuse_trees:
            if not surpress_warnings: warnings.warn("Not reusing the previously built trees during update!", stacklevel=2)
        
        self.fully_observable = fully_observable
        if self.fully_observable:
            if not surpress_warnings: warnings.warn("Running in fully observable mode. Assuming state is observable and we don't need to sample.", stacklevel=2)

        self.max_plus_iterations = max_plus_iterations

        self.C = 10 # PFT constant

        self.use_varel = use_varel
        if isinstance(graph, CoordinationGraph) and self._num_factors > 1:
            if not self._num_factors == graph.num_edges:
                if not self.FACTORED_STATISTICS:
                    # if not surpress_warnings: 
                    warnings.warn("There should be a factor for every edge in the coordination graph!")
            for factor, (i, j) in zip(self._factors, graph.edges):
                factor.agent_ids = [i, j]
            self.elimination_tree = EliminationTree(graph)
        else:
            assert not self.use_varel, "Can't use variable elimination without defining a graph."
            assert self.num_trees == 1, "Define graph to run with multiple factors."

        self.likelihood_sampling = likelihood_sampling
        self.weighted_pf = weighted_particle_filtering
        self.PARTICLE_CLS = WPF if self.weighted_pf else Particles
        self.factored_filter = factored_pf
        self.use_particles = use_particles
        if self.use_particles:
            if self.fully_observable:
                if not surpress_warnings: warnings.warn("Don't use particles when the system is fully observable.")
            max_particles = num_particles # 5*num_particles
            self.particle_manager : ParticleManager = ParticleManager(self.simulator, minimum_n_particles=20, num_particles=num_particles, weighted=self.weighted_pf, max_num_particles=max_particles)
            self.n_particles = num_particles
            for i in range(self.num_trees):
                # Add the initial random particle filter for all factors individiually.
                self._factors[i].belief = self.particle_manager.build_initial_belief(num_particles)
            self.global_particles = global_particles
            if self.global_particles:
                self.global_particles_belief = self.particle_manager.build_initial_belief(num_particles)

            if self.factored_filter:
                if self.FACTORED_STATISTICS:
                    self.particle_manager.state_indices = [list(self.simulator.get_state_indices(e[0]).union(self.simulator.get_state_indices(e[1]))) for e in graph.edges]
                elif self.FACTORED_TREES:
                    self.particle_manager.state_indices = [list(self.simulator.get_state_indices(f.agent_ids[0]).union(self.simulator.get_state_indices(f.agent_ids[1]))) for f in self._factors]
                else:
                    raise ValueError("Should not happen.")
                self.factored_filter_belief = self.particle_manager.build_initial_belief(num_particles)

        self.action_coordination_debug = False # Heavy flag, will compute both VarEl and MaxPlus when computing actions and outputs their values if not equal.
        if self.action_coordination_debug:
            warnings.warn("Test run enabled. Computing both VarEl and MaxPlus outputs.")

        self.graph = graph

        # Dirichtlet noise factor for exploration
        self.alpha = 10 / 9

        # Use monte carlo particles?
        self.use_simulated_particles = use_sim_particles # not self.weighted_pf

        # Default uniform prior.
        self.prior = None

        self.DPW = progressive_widening
        self.POMCPOW = pomcpow
        self.POMCPOW_GLOBAL = pomcpow_global
        assert not (self.POMCPOW_GLOBAL and self.FLAT_POMCP), "Use default POMCPOW with single tree."
        self.PFT = pft
        assert not (self.DPW and self.POMCPOW), "Choose either DPW or POMCPOW versions of the algorithm."

        # For the non-standard simulate functions:
        self.reward_map = {}
        self.k_o = self.k_a = 5
        self.a_o = self.a_a = 1 / 20

        self.do_asymmetric_sampling = asymmetric_sampling

        assert not (self.factored_filter and self.do_asymmetric_sampling) and not(self.do_asymmetric_sampling and self.likelihood_sampling)

        self.single_tree = self.num_trees == 1

        self.joint_history = []

        self.belief_dict = {}

        if self.FACTORED_STATISTICS: 
            assert self.single_tree, "Factored Statistics is implemented only to work with Flat POMCP (i.e. 1 factor a.k.a. tree)."
        
        self.debug = {}

    def _get_possible_actions(self, factored) -> range:
        return self.simulator.get_possible_actions(factored)
    
    def _get_rollout_action_from_simulator(self, state):
        return self.simulator.get_random_action(state)

    def action_size(self, agent : int, *args, **kwargs):
        return self.simulator.action_size(agent)

    def plan(self, belief : list = None, true_state = None):
        return self._SEARCH(None, belief, asymmetric_sampling=self.do_asymmetric_sampling, true_state=true_state)
    
    def uniform_sampling_with_global_fallback(self, random_samples : np.ndarray, current_n_sims : int):
        """
        Uniform or biased roottree sampling of the factored particle filters.
        """
        designated_factor = self._factors[random_samples[current_n_sims]]
        sample = False
        available_factors = set(range(self.num_trees))
        while not sample:
            b = designated_factor.belief
            if len(b) != 0:
                sample = True
            else:
                available_factors -= {designated_factor.factor_id}
            if len(available_factors) == 0:
                # Fallback to global filter.
                warnings.warn("Sampling from global particle filter!")
                if self.global_particles:
                    b = self.global_particles_belief
                else:
                    raise ParticleFilterException("Nothing to sample from.")
                sample = True
            else:
                designated_factor = self._factors[random.choice(list(available_factors))]
        state = b.sample()
        assert state is not None, (state, designated_factor, b, available_factors)
        return state

    def check_beliefs(self, beliefs):
        if not any([len(b) > 0 for b in beliefs]):
            raise ParticleFilterException("All empty beliefs.")

    def get_sim_state(self, beliefs, asymmetric_sampling, random_samples, likelihoods, n_sims):
        if self.use_particles:
            if self.factored_filter:
                state = self.factored_filter_belief.sample()
            elif self.likelihood_sampling:
                self.check_beliefs(beliefs)
                if sum(likelihoods) == 0:
                    # Resort to uniform if likelihoods are too small.
                    state = random.choice(beliefs).sample()
                else:
                    state = random.choices(beliefs, weights=likelihoods, k=1)[0].sample()
            else:
                state = self.uniform_sampling_with_global_fallback(random_samples, n_sims)
        else:
            # Use a state sampled from the joint belief.
            state = random_samples[n_sims]
        return state
    
    def get_PFT_initial_state(self, belief):
        return WPF([(belief.sample(), 1 / self.C) for _ in range(self.C)])

    # @profile(sort='tottime')
    def _SEARCH(self, history : list[tuple], belief : list, real_observation : int = -1, real_action : int = -1, asymmetric_sampling = False, true_state = None):
        n_sims = 0
        returns = []
        depths = []
        self.max_reached_depth = 0
        t_elapsed = 0

        if not self.fully_observable:
            if self.use_particles:
                if not asymmetric_sampling:
                    # Precompute the independent random samples using numpy for efficiency.
                    # Decide uniformly which factor to sample from.
                    random_samples = np.random.randint(0, self.num_trees, size=self._num_sims)
                else:
                    random_samples = None
            else:
                # Sample states to run simulations on from the joint belief. 
                random_samples = np.random.choice(range(self.simulator.get_num_states()), p=belief, size=self._num_sims)
        if self.use_particles:
            # if self.PFT:
            beliefs = [f.belief for f in self._factors if len(f.belief) > 0 and (not self.weighted_pf or sum(f.belief.weights) > 0)] # FIXME what to do with all 0 particles? Sysadmin?
            if len(beliefs) == 0:
                raise ParticleFilterException("No beliefs to sample from!")
            # else:
                # beliefs = [f.tree.belief if f.tree is not None else f.belief for f in self._factors]
            likelihoods = None
            if self.likelihood_sampling:
                likelihoods = [b.get_likelihood() for b in beliefs]
            if self.global_particles:
                beliefs.append(self.global_particles_belief)

        while True:
            # Assign assumed state
            try:
                if self.PFT:
                    # state = self.get_PFT_initial_state([f.tree.belief if f.tree is not None else f.belief for f in self._factors][0]) # TODO
                    if self.fully_observable:
                        state = WPF([(copy.deepcopy(true_state), 1 / self.C) for _ in range(self.C)])
                    elif self.likelihood_sampling:
                        self.check_beliefs(beliefs)
                        if sum(likelihoods) == 0:
                            # Resort to uniform if likelihoods are too small.
                            belief = random.choice(beliefs)
                        else:
                            belief = random.choices(beliefs, weights=likelihoods, k=1)[0]
                        state = self.get_PFT_initial_state(belief)
                    else:
                        state = self.get_PFT_initial_state(random.choice(beliefs))
                elif self.fully_observable:
                    if true_state is not None:
                        state = copy.deepcopy(true_state)
                    else:
                        # Hacky way to get the initial state of the underlying simulator.
                        state = self.simulator._simulator.state
                else:
                    state = self.get_sim_state(beliefs, asymmetric_sampling, random_samples, likelihoods, n_sims)
            except ParticleFilterException as ve:
                # print([str(b) for b in beliefs], [str(f.belief) for f in self._factors])
                raise ve            
            
            simulate = self.SIMULATE_PFT_TRUE if self.PFT else self._SIMULATE_DPW if self.DPW else self.SIMULATE_POMCPOW_REAL if self.POMCPOW else self.SIMULATE_POMCPOW_GLOBAL if self.POMCPOW_GLOBAL else self._SIMULATE
            tik = time.time()

            r = simulate(
                state, 
                # history, 
                [self._factors[i].tree for i in range(self.num_trees)], 
                [None for _ in range(self.num_trees)],
                [(real_observation) for _ in range(self.num_trees)], # INITIAL OBSERVATIONS?
                [(real_action) for _ in range(self.num_trees)], # INITIAL Actions?
                0)

            tok = time.time()
            t_elapsed += tok - tik
            # These estimates are not really used for anything other than debugging.
            returns.append(r)

            depths.append(self.max_reached_depth)
            self.max_reached_depth = 0

            for i in range(self.num_trees): assert self._factors[i].tree is not None

            n_sims += 1

            if n_sims >= self._num_sims or t_elapsed > self._max_time:
                break

        trees = [f.tree for f in self._factors]

        # For the best action we actually need to do a final run of action selection (maxplus or other).
        if self.FACTORED_TREES or self.FACTORED_STATISTICS:
            # Final action selection should not incorporate exploration bonus.
            best_action = self.select_action(trees, use_ucb=False)
        else:
            # If only one tree and no factored statistics, we're running joint so just use argmax.
            best_action = trees[0].argmax()
        
        visiteds = [x.tree.num_visits if x.tree is not None else 0 for x in self._factors]
        total_visited = sum(visiteds)

        assert total_visited / self.num_trees >= n_sims-100, (total_visited, n_sims, visiteds)

        return best_action, trees, t_elapsed, n_sims, returns, depths
    
    def G(self, s, a):
        return self.simulator.generative_transition(s, a)

    def _sample_model(self, s, a):
        next_s, o, r, n = self.G(s, [a])
        return next_s, o, r, n
    
    def belief_to_rollout(self, sw, depth):
        s, w = sw
        return w*self._ROLLOUT(s, depth)
    
    def _ROLLOUT_PFT(self, belief, depth):
        V = 0
        for (s, w) in belief:
            V += w * self._ROLLOUT(s, depth)
        return V

    """
    SparsePFT variant's SIMULATE procedure, contains the procedure for both Sparse-PFT, FS-PFT, and FT-PFT.
    """
    def SIMULATE_PFT_TRUE(self, belief : WPF, roots : list[VNode], parents : list[QNode], observations : list[tuple], actions : list[tuple], depth : int, rho : float = None):
        if depth > self._max_depth:
            return 0
        self.max_reached_depth = max(self.max_reached_depth, depth)

        for i in range(self.num_trees):
            if roots[i] is None:
                if self._factors[i].tree is None:
                    roots[i] = self._construct_VNode(self._factors[i], root=True)
                    self._factors[i].tree = roots[i]
                else:
                    roots[i] = self._construct_VNode(self._factors[i])
                self._expand_VNode(roots[i], prior=None, value=self._value_init)
                # roots[i].belief = belief
                # roots[i].rho = rho

        joint_action = self.select_action(roots, use_ucb=True, progressive_widening=self.DPW)
        factored_joint_action = self.check_and_fix_factored_actions(joint_action)
    
        if self.FACTORED_STATISTICS:
            assert len(roots) == 1 and self.single_tree, "There should be one tree."
            if roots[0][factored_joint_action[0]] is None:
                roots[0][factored_joint_action[0]] = QNode(self._num_visits_init, self._value_init, parent=roots[0])

        expand, o, has = [None] * self.num_trees, [None] * self.num_trees, [None] * self.num_trees
        rhos, beliefs, nodes = [], [], []

        for i in range(self.num_trees):
            has[i] = roots[i][factored_joint_action[i]]
            o[i] = len(has[i].children)
            expand[i] = o[i] < len(belief) / self.num_trees

        terminal = False

        if any(expand):
            belief_, rho_, _, terminal = self.particle_manager.G_PF(belief, joint_action)
            if not terminal:
                for i in range(self.num_trees):
                    if roots[i][factored_joint_action[i]][o[i]] is None:
                        roots[i][factored_joint_action[i]][o[i]] = self._construct_VNode(self._factors[i])
                        self._expand_VNode(roots[i][factored_joint_action[i]][o[i]], prior=None, value=self._value_init)
                        roots[i][factored_joint_action[i]][o[i]].belief = belief_
                        roots[i][factored_joint_action[i]][o[i]].rho = rho_
                    else:
                        if self.num_trees == 1:
                            raise ValueError("")
        else:
            for i in range(self.num_trees):
                ha = has[i]
                if o[i] <= 0:
                    raise ValueError("Should not occur; there are no children.")
                if o[i] > 1:
                    w_o = list(map(lambda x : x.num_visits+1, ha.children.values()))
                    o[i] = random.choices(list(ha.children.keys()), weights=w_o, k=1)[0]
                else:
                    o[i] = list(ha.children.keys())[0]
                node = ha[o[i]]
                nodes.append(node)
        
        rollout = any([roots[i].num_visits <= 0 for i in range(self.num_trees)])

        if len(nodes) > 0:
            node = random.choice(nodes)
            belief_ = node.belief
            rho_ = node.rho

        if terminal:
            total_reward = rho_
        elif rollout:
            total_reward = rho_ + self._discount_factor * self._ROLLOUT_PFT(belief_, depth+1)
        else:
            total_reward = rho_ + self._discount_factor * self.SIMULATE_PFT_TRUE(belief_, 
                                                    [roots[i][factored_joint_action[i]][o[i]] for i in range(self.num_trees)],
                                                    [roots[i][factored_joint_action[i]] for i in range(self.num_trees)],
                                                    o, factored_joint_action, depth+1, rho_)

        self.update_statistics(roots, factored_joint_action, joint_action, None, o, total_reward)

        return total_reward

    def _SIMULATE_DPW(self, state : int, roots : list[VNode], parents : list[QNode], observations : list[tuple], actions : list[tuple], depth : int):
        if depth > self._max_depth:
            return 0
        self.max_reached_depth = max(self.max_reached_depth, depth)
        rollout = [False] * self.num_trees
        for i in range(self.num_trees):
            if roots[i] is None:
                rollout[i] = True
                if self._factors[i].tree is None:
                    roots[i] = self._construct_VNode(self._factors[i], root=True)
                    self._factors[i].tree = roots[i]
                else:
                    roots[i] = self._construct_VNode(self._factors[i])
                self._expand_VNode(roots[i], prior=None, value=self._value_init)

        joint_action = self.select_action(roots, use_ucb=True, progressive_widening=True)
        next_state, new_observations, reward, nsteps = self.G(state, joint_action)
        terminal_state = nsteps == 0
        factored_joint_action = self.check_and_fix_factored_actions(joint_action)
        self.reward_map[(tuple(state), tuple(joint_action), tuple(next_state))] = reward
        next_state_set = set()
        for i in range(self.num_trees):
            ha = roots[i][factored_joint_action[i]]
            os = len(ha.children)
            # Progressive Widening in Observation Space
            if os <= self.k_o * (ha.num_visits ** self.a_o):
                ha[new_observations[i]] = self._construct_VNode()
                # ha[new_observations[i]].belief.add(next_state)
                self.add_particle(ha[new_observations[i]].belief, next_state, joint_action, new_observations, new_observations[i], self._factors[i].agent_ids)
                self._expand_VNode(ha[new_observations[i]]) # TODO: state?
                # rollout[i] = True
            else:
                # new_observations[i] = list(ha.children.keys())[random.randint(0, os-1)]
                if os > 1:
                    w_o = np.array(list(map(lambda x : x.num_visits, ha.children.values())))
                    new_observations[i] = random.choices(list(ha.children.keys()), weights=(w_o / w_o.sum()).tolist(), k=1)[0]
                else:
                    new_observations[i] = list(ha.children.keys())[0]
                next_state_set.union(set(ha[new_observations[i]].belief.values))
                rollout[i] = False
        
        if len(next_state_set) > 0:
            next_state = random.choice(next_state_set)
            reward = self.reward_map.get((tuple(state), tuple(factored_joint_action), tuple(next_state)), None)
            if reward is None:
                if hasattr(self.simulator, 'reward_model'):
                    reward = self.simulator.reward_model(state, joint_action, next_state)
                else:
                    raise ValueError("=> (s,a,s') not seen earlier and no reward model specified.")

        if any(rollout):
            return reward + self._discount_factor * self._ROLLOUT(next_state, depth+1, actions if depth > 0 else None)

        if terminal_state:
            total_reward = reward
        else:
            total_reward = reward + self._discount_factor * self._SIMULATE_DPW(next_state, 
                                                [roots[i][factored_joint_action[i]][new_observations[i]] for i in range(self.num_trees)],
                                                [roots[i][factored_joint_action[i]] for i in range(self.num_trees)],
                                                new_observations, factored_joint_action, depth+nsteps)

        self.update_statistics(roots, factored_joint_action, joint_action, state, new_observations, total_reward)

        return total_reward

    """
    The SIMULATE function for (W-)POMCP variants. Supports runn with both factored statistics as well as factored trees.
    """
    def _SIMULATE(self, state : int, roots : list[VNode], parents : list[QNode], observations : list[tuple], actions : list[tuple], depth : int):
        if depth > self._max_depth:
            return 0
        self.max_reached_depth = max(self.max_reached_depth, depth)
        rollout = [False] * self.num_trees
        for i in range(self.num_trees):
            if roots[i] is None:
                vs = []
                rollout[i] = True
                if self._factors[i].tree is None:
                    roots[i] = self._construct_VNode(self._factors[i], root=True)
                    self._factors[i].tree = roots[i]
                else:
                    roots[i] = self._construct_VNode(self._factors[i])

                if parents[i] is not None:
                    parents[i][observations[i]] = roots[i]

                if rollout[i]:
                    v = self._value_init
                    pis = self.prior
                    self._expand_VNode(roots[i], prior=pis, value=v)
                    vs.append(v)

        if any(rollout):
            return self._ROLLOUT(state, depth, actions if depth > 0 else None)

        joint_action = self.select_action(roots, use_ucb=True)
        next_state, new_observations, reward, nsteps = self.G(state, joint_action)

        terminal_state = nsteps == 0

        factored_joint_action = self.check_and_fix_factored_actions(joint_action)
        assert len(factored_joint_action) == self.num_trees == len(new_observations), \
                (factored_joint_action, len(factored_joint_action), new_observations, len(new_observations), self.num_trees)

        if self.FACTORED_STATISTICS:
            assert len(roots) == 1 and self.single_tree, "There should be one tree."
            if roots[0][factored_joint_action[0]] is None:
                roots[0][factored_joint_action[0]] = QNode(self._num_visits_init, self._value_init, parent=roots[0])

        if terminal_state:
            total_reward = reward
        else:
            total_reward = reward + self._discount_factor * self._SIMULATE(next_state,
                                                                               [roots[i][factored_joint_action[i]][new_observations[i]] for i in range(self.num_trees)],
                                                                               [roots[i][factored_joint_action[i]] for i in range(self.num_trees)],
                                                                               new_observations,
                                                                               factored_joint_action,
                                                                               depth+nsteps)

        # The pseudocode says we add s to the particle filters here
        self.update_statistics(roots, factored_joint_action, joint_action, state, new_observations, total_reward)

        return total_reward

    """
    Update the statistics for a tree nodes; visit counts, Q-value estimates, and, for POMCP, particles.
    """
    def update_statistics(self, roots : list[VNode], factored_joint_action, joint_action : list, state, joint_factored_observations : list, total_reward : float) -> None:
        if self.FACTORED_STATISTICS:
            # Only factored statistics / value, single tree.
            assert len(roots) == 1, "There should be one root in factored statistics."
            rootnode = roots[0]
            factored_joint_action = FactoredSimulators.total_to_factored(self.graph, joint_action)
            for i, a in enumerate(factored_joint_action):
                statistic = rootnode.factored_statistics[i][a]
                statistic.num_visits += 1
                statistic.value = statistic.value + ((total_reward - statistic.value) / (statistic.num_visits))
                rootnode.factored_statistics[i][a] = statistic
            rootnode.num_visits += 1
            if state is not None:
                if self.use_particles and not (self.DPW or self.POMCPOW or self.PFT) and not isinstance(rootnode, RootVNode) and self.use_simulated_particles:
                    self.add_particle(rootnode.belief, state, joint_action, joint_factored_observations, joint_factored_observations[0], self._factors[0].agent_ids)
        else:
            # Factored Trees, which also results in factored statistics by itself.
            for i in range(self.num_trees):
                action = factored_joint_action[i]
                if state is not None:
                    if self.use_particles and not (self.DPW or self.POMCPOW or self.PFT) and not isinstance(roots[i], RootVNode) and self.use_simulated_particles:
                        self.add_particle(roots[i].belief, state, joint_action, joint_factored_observations, joint_factored_observations[i], self._factors[i].agent_ids)
                roots[i].num_visits += 1
                roots[i][action].num_visits += 1
                roots[i][action].value = roots[i][action].value + ((total_reward - roots[i][action].value) / (roots[i][action].num_visits))
            if self.use_particles and self.global_particles:
                self.add_particle(self.global_particles_belief, state, joint_action, joint_factored_observations)

    def add_particle(self, belief : Particles, state, joint_action : list, joint_factored_observations : list, factored_obs = None, agent_ids = None, force_weighted = False) -> None:
        if self.weighted_pf or force_weighted:
            assert belief.__class__ in {WPF, WeightedParticles}
            if self.FACTORED_TREES and hasattr(self.simulator, 'factored_obs_prob'):
                w = self.simulator.factored_obs_prob(state, joint_action, factored_obs, agent_ids)
            else:
                w = self.simulator.obs_prob(state, joint_action, joint_factored_observations)
            belief.add((state, w))
        else:
            belief.add(state)            

    """
    Rollout function to determine the value of a new node.
    """
    def _ROLLOUT(self, state, depth : int, action = None, use_heuristics = True):
        if depth == 0: # Rollouts at the first layer are worthless as the result is discarded.
            return 0
        discount = 1.0
        total_discounted_reward = 0
        nsteps = 1
        while depth < self._max_depth:
            if use_heuristics:
                if action is None:
                    # Pass None to get random action, pass state to get heuristic action.
                    action = self._get_rollout_action_from_simulator(None)
                else:
                    action = self._get_rollout_action_from_simulator(state)
            else:
                action = self._get_rollout_action_from_simulator(None)
            next_state, _, reward, nsteps = self.G(state, action)
            depth += nsteps
            total_discounted_reward += reward * discount
            if nsteps == 0:
                break
            discount *= (self._discount_factor**nsteps)
            state = next_state
        return total_discounted_reward

    def replace_graph(self, graph : CoordinationGraph) -> None:
        self.graph = graph
        self.elimination_tree = EliminationTree(graph)

    def replace_simulator(self, sim : Simulator) -> None:
        self.simulator = sim

    """
    Find the Q-values for the edges from the root-nodes in the tree when using factored value estimation.
    """
    def roots_to_edge_qs(self, roots : list[VNode], use_ucb=False, progressive_widening=False):
        def edge_Qs(roottree : VNode, a1 : int, a2 : int):
            as1, as2 = self.action_size(a1), self.action_size(a2)
            # Vanilla value or ucb value
            restrict_widening = False
            if progressive_widening:
                counts = np.fromiter(map(lambda x : x.num_visits, roottree.children.values()), dtype=int, count=as1*as2)
                num_children = counts[counts > 0].size
                if not (num_children <= self.k_a * (roottree.num_visits ** self.a_a)):
                    restrict_widening = True # no widening
            if use_ucb:
                get_value_f = lambda x : -1e3 if restrict_widening else opt_UCB(x.value, roottree.num_visits, x.num_visits, self.c) 
            else:
                get_value_f = lambda x : -1e3 if restrict_widening else x.value
            try:
                # Compute edge/factor Q value. This is the combined Q-function conditioned on the actions of both agents, Q( - | a1, a2 ).
                assert self.FACTORED_TREES, "We should only be able to reach this if the algorithm is running factored trees."
                # edge_factor_Q_dist = np.array(list(map(get_value_f, roottree.children.values()))).reshape(self.action_size(a1), self.action_size(a2))
                edge_factor_Q_dist = np.fromiter(map(get_value_f, roottree.children.values()), dtype=float, count=as1*as2).reshape(as1, as2)
            except ValueError as ve:
                print(roottree)
                raise(ve)
            return edge_factor_Q_dist

        if self.FACTORED_STATISTICS:
            node = roots[0]
            edge_factor_Q_dists = [None] * self._num_factors
            for edge_id, factor_stats in enumerate(node.factored_statistics):
                a1, a2 = self.graph.edges[edge_id]
                edge_qs = np.array(list(map(lambda x : self.UCB_on_stats(x.value, node.num_visits, x.num_visits), factor_stats.values()))).reshape(self.action_size(a1), self.action_size(a2))
                edge_factor_Q_dists[edge_id] = edge_qs
            edge_factor_Q_dists = np.array(edge_factor_Q_dists)
        else:
            edge_factor_Q_dists = np.array([edge_Qs(tree, *self.graph.edges[edge_id]) for edge_id, tree in enumerate(roots)])
        return edge_factor_Q_dists

    """
    Select UCB1 or greedy action.
    """
    def select_action(self, roots : list[VNode], naive=False, **kwargs):
        if naive or self.FLAT_POMCP:
            action_selection = self._ucb_ma_naive
        else:
            if self.use_varel:
                action_selection = partial(self.varel, **kwargs)    
            else:
                action_selection = partial(self.max_plus, **kwargs)
        return action_selection(roots=roots)

    """
    Variable elimination.
    """
    def varel(self, roots : list[VNode], **kwargs):
        edge_qs = self.roots_to_edge_qs(roots, **kwargs)
        edge_q_dict = {}
        graph = self.graph
        el_tree = self.elimination_tree
        for idx, arr in enumerate(edge_qs):
            edge_q_dict[idx] = arr
        result = el_tree.agent_elimination(edge_q_dict, **kwargs)
        return np.array(result)

    """
    Max-Plus. Requires both UCB1 and 'vanilla' Q-values.
    """
    def max_plus(self, roots : list[VNode], **kwargs):
        edge_Q, edge_Q_UCB = self.roots_to_edge_qs(roots, use_ucb=False), self.roots_to_edge_qs(roots, use_ucb=True)
        graph = self.graph
        return max_plus_ucb_final(graph, self.max_plus_iterations, edge_Q, edge_Q_UCB, self.action_size, agent_ordering=self.elimination_tree.agentorder, use_ucb=kwargs['use_ucb'])

    def _ucb_ma_naive(self, roots : list[VNode]):
        """
        Run UCB on every factor independently. In practice only used for single trees/flat pomcp but it's compatible with multiple.
        """
        best_action = [None for _ in range(self.num_trees)]
        for i in range(self.num_trees):
            best_action[i] = self._ucb(roots[i])
        assert None not in best_action
        return best_action

    def _ucb(self, root : VNode):
        """
        UCB (upper confidence bound) action selection function for a single tree.
        """
        best_action = []
        best_value = float('-inf')
        for action, qnode in root.children.items():
            if qnode.num_visits == 0:
                value = float('inf')
            else:
                value = self.UCB(root, qnode)
            if value > best_value:
                best_action = [action]
                best_value = value
            elif math.isclose(value, best_value):
                best_action.append(action)
        # Random tiebreaks
        return best_action[0] if len(best_action) == 1 else random.choice(best_action)


    """
    Helper method to update PFT search trees and beliefs.
    """
    def update_PFT(self, factor : Factor, real_action, real_observation, action_individual_indices : list[int]) -> None:
        if real_action not in factor.tree or real_observation not in factor.tree[real_action]:
            factor.belief = self.particle_manager.update_particles(factor.belief, action_individual_indices, real_observation, factor=(factor if self.num_trees > 1 else None), reinvigoration=True)
            factor.tree = None
        else:
            # factor.belief = factor.tree.belief
            factor.belief = self.particle_manager.update_particles(factor.belief, action_individual_indices, real_observation, factor=(factor if self.num_trees > 1 else None), reinvigoration=True)
            factor.tree = RootVNode.from_vnode(factor.tree[real_action][real_observation], factor.history)

    """
    Helper method for updating search trees and beliefs.
    """
    def update_(self, factor : Factor, real_action, real_observation, action_individual_indices : list[int]) -> None:
        """
        Assume that the tree's history has been updated after taking real_action and receiving real_observation.
        """
        if self.PFT:
            return self.update_PFT(factor, real_action, real_observation, action_individual_indices)
        if real_action not in factor.tree or real_observation not in factor.tree[real_action]: # len(factor.tree[real_action].children) == 0:
            if self.use_particles: 
                factor.belief = self.particle_manager.update_particles(factor.belief, action_individual_indices, real_observation, factor=(factor if self.num_trees > 1 else None), reinvigoration=True)
            # This factor tree has to start from scratch.
            factor.tree = None
            return

        if factor.tree[real_action][real_observation] is not None:
            if self.use_particles:
                updated_particles = self.particle_manager.update_particles(factor.belief, action_individual_indices, real_observation, factor=(factor if self.num_trees > 1 else None), reinvigoration=False)
            # Prune tree
            new_node = RootVNode.from_vnode(factor.tree[real_action][real_observation], factor.history)
            if self.use_particles and updated_particles:
                if self.use_simulated_particles:
                    new_belief = self.particle_manager.merge_simulation_and_updated_particles(new_node.belief, updated_particles)
                else:
                    new_belief = updated_particles
                    new_belief = self.particle_manager.particle_reinvigoration(factor.belief, new_belief)
                factor.belief = new_belief
            factor.tree = new_node
        else:
            raise ValueError("Unexpected state, child can't be None. This means we never expected/simulated the received observation in the tree, which we can't handle (yet).")
    
    def check_and_fix_factored_actions(self, actions):
        if self.FACTORED_TREES:
            return FactoredSimulators.total_to_factored(self.graph, actions)
        if self.FACTORED_STATISTICS:
            return [tuple(actions)]
        return actions

    """
    The method to call for updating search trees and beliefs.
    """
    def update(self, real_actions, real_observations):
        # Append (a, o) to joint history (keep in mind, o is factored).
        self.joint_history.append((real_actions, real_observations))

        factored_actions = self.check_and_fix_factored_actions(real_actions)
        for i in range(self.num_trees):
            # Append (ae, oe) to factor history.
            self._factors[i].history.append((factored_actions[i], real_observations[i]))
            self.update_(self._factors[i], factored_actions[i], real_observations[i], action_individual_indices=real_actions)
            if self.dont_reuse_trees:
                # If we don't reuse previously built trees, we force a rebuild by setting the trees of the factors to None.
                self._factors[i].tree = None

        if self.use_particles and self.factored_filter:
            assert self.factored_filter_belief is not None
            t_obs = FactoredSimulators.factored_to_total(self.graph, real_observations) if self.FACTORED_TREES else real_observations
            self.factored_filter_belief = self.particle_manager.update_factored_particles(self.factored_filter_belief, real_actions, t_obs, 
                                                                                          FactoredSimulators.total_to_factored(self.graph, real_actions),
                                                                                          real_observations,
                                                                                          self._num_factors)
        # Update global particle filter:
        # real_actions is a flat list of action per agent.
        # real observations can be a factored or flat list of observations. 
        # However in the former case, then this same simulator will return factored observations in the rejection algorithm as well.
        if self.use_particles and self.global_particles:
            self.global_particles_belief = self.particle_manager.update_particles(self.global_particles_belief, real_actions, real_observations)

    def _construct_VNode(self, factor : Factor = None, root=False, **kwargs):
        """
        Construct a (root) VNode for this factor tree and initialise its values.
        """
        vnode = RootVNode(self._num_visits_init, None) if root else VNode(self._num_visits_init)

        if self.use_particles:
            vnode.belief = self.PARTICLE_CLS([])

        return vnode

    def _expand_VNode(self, vnode : VNode, prior=None, value=None):
        if self.FACTORED_STATISTICS:
            # vnode.factored_statistics = [defaultdict(FactoredStatistic) for _ in range(self._num_factors)]
            vnode.factored_statistics = [{a : FactoredStatistic() for a in self._get_possible_actions(factored=True)} for _ in range(self._num_factors)]
            return
        for action_idx, action in enumerate(self._get_possible_actions(factored=(not self.single_tree))):
            if vnode[action] is None:
                if prior is not None and prior[action_idx] is not None:
                    qnode = QNode(self._num_visits_init, value if value else self._value_init, prob=prior[action_idx], parent=vnode)
                    vnode[action] = qnode
                else:
                    qnode = QNode(self._num_visits_init, value if value else self._value_init, parent=vnode)
                    vnode[action] = qnode

    def UCB(self, root : VNode, child : QNode) -> float:
        """
        Upper Confidence Bound for Upper Confidence Trees algorithm.
        """
        return self.UCB_on_stats(child.value, root.num_visits, child.num_visits)
        # return child.value + self.c * math.sqrt(math.log(root.num_visits + 1) / (child.num_visits + 1))
    
    def UCB_on_stats(self, value, root_visits, child_visits) -> float:
        return opt_UCB(value, root_visits, child_visits, self.c)
