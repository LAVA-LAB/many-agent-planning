from __future__ import annotations
from collections import defaultdict

import random
import copy
from typing import Union
import warnings
from functools import reduce
import math

import pandas as pd
from envs.factored_simulator_helper import FactoredSimulators

from envs.simulator import Simulator

import numpy as np
import scipy.stats as scstats

class ParticleFilterException(Exception):
    pass

"""
Credits for WeightedParticles class due to Kaiyu Zheng: https://github.com/h2r/pomdp-py/blob/master/pomdp_py/representations/distribution/particles.pyx
"""

"""
Class to manage weighted particles (both online and offline).
"""
class WeightedParticles:
    """
    Represents a distribution :math:`\Pr(X)` with weighted particles, each is a
    tuple (value, weight). "value" means a value for the random variable X. If
    multiple values are present for the same value, will interpret the
    probability at X=x as the average of those weights.
    __init__(self, list particles, str approx_method="none", distance_func=None)
    Args:
       particles (list): List of (value, weight) tuples. The weight represents
           the likelihood that the value is drawn from the underlying distribution.
       approx_method (str): 'nearest' if when querying the probability
            of a value, and there is no matching particle for it, return
            the probability of the value closest to it. Assuming values
            are comparable; "none" if no approximation, return 0.
       distance_func: Used when approx_method is 'nearest'. Returns
           a number given two values in this particle set.
    """
    def __init__(self, particles, approx_method="none", distance_func=None):
        self._values = [value for value, _ in particles]
        self._weights = [weight for _, weight in particles]
        self._particles = particles

        # self._hist = self.get_histogram()
        self._hist_valid = False

        self.weighted = True

        self._approx_method = approx_method
        self._distance_func = distance_func

    @property
    def particles(self):
        return self._particles

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        return self._weights

    def add(self, particle):
        """add(self, particle)
        particle: (value, weight) tuple"""
        assert particle is not None
        self._particles.append(particle)
        s, w = particle
        self._values.append(s)
        self._weights.append(w)
        self._hist_valid = False

    def __str__(self):
        return str(sorted(self.condense().particles, key=lambda x : x[1], reverse=True))

    def __len__(self):
        return len(self._particles)

    def __getitem__(self, value):
        """Returns the probability of `value`; normalized"""
        if len(self.particles) == 0:
            raise ParticleFilterException("Particles is empty.")

        if not self._hist_valid:
            self._hist = self.get_histogram()
            self._hist_valid = True

        if value in self._hist:
            return self._hist[value]
        else:
            if self._approx_method == "none":
                return 0.0
            elif self._approx_method == "nearest":
                nearest_dist = float('inf')
                nearest = self._values[0]
                for s in self._values[1:]:
                    dist = self._distance_func(s, nearest)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = s
                return self[nearest]
            else:
                raise ParticleFilterException("Cannot handle approx_method:",
                                 self._approx_method)

    def __setitem__(self, value, prob):
        """
        The particle belief does not support assigning an exact probability to a value.
        """
        raise NotImplementedError

    def random(self):
        """Samples a value based on the particles"""
        value = random.choices(self._values, weights=self._weights, k=1)[0]
        return value
    
    def sample(self):
        """Alias for random above. Only return the state, not the weight."""
        if len(self) > 0:
            return self.random()
        else:
            raise ParticleFilterException("Sampling from deprived filter!")

    def mpe(self):
        if not self._hist_valid:
            self._hist = self.get_histogram()
            self._hist_valid = True
        return max(self._hist, key=self._hist.get)

    def __iter__(self):
        return iter(self._particles)
    
    def normalise(self):
        assert None not in self._weights, 'Normalising unweighted particle filter?'
        total_weight = sum(self.weights)
        self._weights = [x / total_weight for x in self._weights]

    def get_histogram(self):
        """
        get_histogram(self)
        Returns a mapping from value to probability, normalized."""
        hist = {}
        counts = {}
        # first, sum the weights
        for s, w in self._particles:
            hist[s] = hist.get(s, 0) + w
            counts[s] = counts.get(s, 0) + 1
        # then, average the sums
        total_weights = 0.0
        for s in hist:
            hist[s] = hist[s] / counts[s]
            total_weights += hist[s]
        # finally, normalize
        for s in hist:
            hist[s] /= total_weights
        return hist
    
    def get_entropy(self):
        if self.weighted:
            probs = self._weights
        else:
            if len(self) in {0, 1}:
                return 0.
            if not self._hist_valid:
                self._hist = self.get_histogram()
                self._hist_valid = True
            probs = list(self._hist.values())
        # Scipy was proven fastest in a very small-scale timeit evaluation.
        return self.get_entropy_scipy(probs)
    
    def get_entropy_scipy(self, probs):
        # Extract the counted or weighted probabilities.
        return scstats.entropy(probs)
    
    def get_entropy_np(self):
        # Extract the counted or weighted probabilities.
        ps = np.array(list(self._hist.values()))
        return - (ps * np.log(ps)).sum()

    @classmethod
    def from_histogram(cls, histogram):
        """
        Given a histogram dictionary, return a particle representation of it, which is an approximation
        """
        particles = []
        for v in histogram:
            particles.append((v, histogram[v]))
        return WeightedParticles(particles)

    def condense(self):
        """
        Returns a new set of weighted particles with unique values
        and weights aggregated (taken average).
        """
        return WeightedParticles.from_histogram(self.get_histogram())

    @classmethod
    def ESS(cls, particles):
        return 1 / np.square(particles.weights).sum()

    def effective_sample_size(self):
        term = np.square(self.weights).sum()
        if term == 0:
            # the limit of 1/x as x approaches zero from the right is positive infinity.
            return np.inf
        else:
            return 1 / term

    @staticmethod
    def KLDSampleSize(k : int, zeta, eta : float = 0.05):
        """
        Return the minimum sample size in order to achieve an error at most ζ with a 1-η level of confidence according to KLD-Sampling.
        """
        a = (max(2.0, k) - 1.0) / 2.0
        b = 1.0/(a*9.0)
        return (1.0-b+math.sqrt(b)*np.quantile(random.random(), 1-eta))**3.0*a/zeta

class WPF(WeightedParticles):

    def __init__(self, particles, approx_method="none", distance_func=None, likelihood=0):
        super().__init__(particles, approx_method, distance_func)
        self.likelihood = likelihood
        self.log = True
    
    def get_likelihood(self):
        if self.log:
            return math.exp(self.likelihood) 
        else:
            return self.likelihood
    
    def update(self, new_particles, total_weight=1., approx_method="none", distance_func=None):
        return WPF(new_particles, approx_method, distance_func, self.likelihood + math.log(total_weight))
        # return WPF(new_particles, approx_method, distance_func, self.likelihood + math.log(total_weight))

class Particles(WeightedParticles):
    """ 
    Particles is a set of unweighted particles; This set of particles represent
    a distribution :math:`\Pr(X)`. Each particle takes on a specific value of :math:`X`.
    Inherits `WeightedParticles`.
    __init__(self, particles, **kwargs)
    Args:
        particles (list): List of values.
        kwargs: see __init__() of `particles.WeightedParticles`.
    """
    def __init__(self, particles, **kwargs):
        super().__init__(list(zip(particles, [None]*len(particles))), **kwargs)
        self.weighted = False

    def __iter__(self):
        return iter(self.particles)

    def add(self, particle):
        """add(self, particle)
        particle: just a value"""
        self._particles.append((particle, None))
        self._values.append(particle)
        self._weights.append(None)
        self._hist_valid = False

    @property
    def particles(self):
        """For unweighted particles, the particles are just values."""
        return self._values

    def get_abstraction(self, state_mapper):
        """
        feeds all particles through a state abstraction function.
        Or generally, it could be any function.
        """
        particles = [state_mapper(s) for s in self.particles]
        return particles

    @classmethod
    def from_histogram(cls, histogram, num_particles=1000):
        """
        Given a histogram dictionary, return a particle representation of it, which is an approximation
        """
        particles = []
        for _ in range(num_particles):
            particles.append(histogram.random())
        return Particles(particles)

    def get_histogram(self):
        hist = {}
        length = len(self.particles)  
        for s in self.particles:
            hist[s] = hist.get(s, 0) + 1
        for s in hist:
            hist[s] = hist[s] / length
        return hist

    def random(self):
        """Samples a value based on the particles"""
        if len(self._particles) > 0:
            return random.choice(self._values)
        else:
            raise ParticleFilterException("Sampling from deprived filter!")

"""
Class to manage the use and updating of particle beliefs \overline{b}.
"""
class ParticleManager:

    def __init__(self, simulator : Simulator, minimum_n_particles : int, num_particles = 20, weighted = False, state_indices : list[list[int]] = None, max_num_particles : int = 2000) -> None:
        self.simulator : Simulator = simulator
        self.minimum_n_particles : int = minimum_n_particles
        self.num_particles : int = num_particles
        self.weighted : bool = weighted
        self.PARTICLE_CLS = WPF if self.weighted else Particles
        self.state_indices : list[list[int]] = state_indices
        self.max_num_particles : int = max_num_particles
        # self.grid : StateGrid = self.simulator.get_state_grid()
        self.delta = 0.3
    
    @staticmethod
    def particle_mapper(f, particles : Union[list, Particles]) -> Particles:
        return Particles([f(p) for p in particles])

    def weighted_particle_reinvigoration(self, particles : WeightedParticles, transform=lambda x : x, ) -> WeightedParticles:
        if len(particles) / particles.effective_sample_size() <= 2:
            return particles
        
        if len(particles) >= self.minimum_n_particles:
            return particles

        new_p = []
        w = 1 / self.minimum_n_particles

        while len(new_p) < self.minimum_n_particles:
            new_p.append((transform(particles.sample()), w))

        return particles.update(new_p)

    def particle_reinvigoration(self, previous_particles : Particles, particles : Particles, transform=lambda x : x, use_previous=False) -> Particles:
        """
        Rebuild particles by adding random samples of the previously observed particles until we have enough particles.
        """
        if self.weighted:
            return self.weighted_particle_reinvigoration(particles, transform)

        if len(particles) == 0: # and len(previous_particles) == 0:
            # What to do, raise an error, return empty particles or rebuild initial belief?
            return particles

        if len(particles) >= self.minimum_n_particles:
            return particles

        while len(particles) < self.minimum_n_particles:
            if use_previous:
                particles.add(previous_particles.random())
                # particles.add(transform(previous_particles.random()))
            else:
                particles.add(particles.random())
                # particles.add(transform(particles.random()))

        return particles
    
    def merge_simulation_and_updated_particles(self, sim_particles : Particles, updated_particles : Particles, transform=None) -> Particles:
        # if self.weighted:
            # wp = WeightedParticles(sim_particles.particles + updated_particles.particles)
            # wp.normalise()
            # return wp

        def get_particle(ps):
            if transform is not None:
                return transform(ps.random())
            else:
                return ps.random()

        if len(sim_particles) == 0:
            return updated_particles
        elif len(updated_particles) == 0:
            return sim_particles
        
        combined_size = len(sim_particles) + len(updated_particles)

        if combined_size < self.max_num_particles:
            if self.weighted:
                wp = WPF(sim_particles.particles + updated_particles.particles)
                wp.normalise()
                return wp
            else:
                return Particles(sim_particles.values + updated_particles.values)

        if len(updated_particles) > self.minimum_n_particles:
            return updated_particles
        else:
            while len(updated_particles) < self.minimum_n_particles:
                updated_particles.add(get_particle(updated_particles))

        if len(sim_particles) + len(updated_particles) == 0:
            return WPF([]) if self.weighted else Particles([])
        
        if self.weighted:
            states = sim_particles.values
            weights = sim_particles.weights

        while len(updated_particles) < self.max_num_particles:
            if self.weighted:
                updated_particles.add(random.choice(states, prob=weights))
            else:
                updated_particles.add(get_particle(sim_particles))

        return updated_particles

    def update_particles(self, particles : Particles, a, o, factor = None, reinvigoration = True) -> Particles:
        if self.weighted:
            return self.update_weighted_belief_by_sir(particles, a, o, factor=factor)

        ps = self.update_belief_by_rejection(particles, a, o, factor=factor)
        if reinvigoration:
            return self.particle_reinvigoration(particles, ps, transform=self.simulator.transform)
        else:
            return ps

    def particle_step(self, s, a):
        s_, sampled_o, r, steps = self.simulator.generative_transition(s, a)
        final_states = 0
        while steps == 0 and final_states < 3:
            s_, sampled_o, r, steps = self.simulator.generative_transition(s, a)
            final_states += 1
        return s_, sampled_o, r, final_states == 3


    def update_weighted_belief_by_sir(self, particles : WeightedParticles, a, o, factor = None) -> WeightedParticles:
        """
        SIR belief update as in the appendix of the AAAI-24 paper.
        """
        assert particles.__class__ in {WeightedParticles, WPF}, ("SIR requires weighted particles!", particles.__class__)
        fp = []
        total_weight = 0
        for s, w in particles:
            s_, _, _, terminal = self.particle_step(s, a)
            if terminal:
                continue
            if factor is not None:
                obs_prob = self.simulator.factored_obs_prob(s_, a, o, factor.agent_ids)
            else:
                obs_prob = self.simulator.obs_prob(s_, a, o)
            # reweighting
            w *= obs_prob
            total_weight += w
            fp.append((s_, w))
        
        if len(fp) == 0:
            return WeightedParticles(fp)

        if total_weight <= 0 or sum(map(lambda x : x[1], fp)) <= 0:
            # We are set to fail
            return particles.update(fp)
        else:
            particles = particles.update(fp, total_weight)
        
        particles.normalise()

        # if particles.effective_sample_size() <= 2:
        if len(particles) / particles.effective_sample_size() <= 2:
        # if particles.effective_sample_size() <= len(particles) / 2:
            # Only resample when necessary.
            # particles.normalise()
            return particles.update(particles, total_weight)
        # RESAMPLING
        sp = []
        for _ in range(len(particles)):
            s = particles.random()
            sp.append((s, 1 / len(particles)))
        return particles.update(sp, total_weight)

    def update_belief_by_rejection(self, particles : Particles, a, o, factor = None) -> Particles:
        """
        Sample random particles from the filter and update by rejection to form the posterior to the encountered action + observation.

        We assume that actions are individual indices and can be passed to the generative environment model as is.
        The observations returned from the simulator are assumed to be factored iff factor_idx is not None.
        """
        fp = []
        max_c = 1
        for s in particles:
        # for _ in range(len(particles)):
            # s = particles.random()
            assert s is not None, particles
            sampled_o = None
            counter = 0
            while sampled_o != o and counter < max_c:
                s_, sampled_o, _, terminal = self.particle_step(s, a)
                if terminal:
                    break
                if factor is not None:
                    # If a factor index is supplied, we assume that the observations are factored and index it.
                    sampled_o = sampled_o[factor.factor_id]
                elif not isinstance(o, list) and isinstance(sampled_o, list) and len(sampled_o) == 1:
                    sampled_o = sampled_o[0] # we're doing normal pomcp
                assert np.array(o).shape == np.array(sampled_o).shape, f"Mismatch in the received observation, {o}, and the observation returned by the environment model, {sampled_o}."
                counter += 1
            if terminal:
                continue
            if sampled_o == o: # or np.mean([math.dist(one, two) for one, two in zip(o, sampled_o)]) < 2:
                # print("SUCCESS!", "STATE:", s, "SAMPLED O:", sampled_o, "NEXT STATE:", s_, "REAL O:", o, "FACTORIDX:", factor_idx, "ACTION:", a, sampled_o != o)
                fp.append(s_)
            else:
                # print("FAILURE!", "STATE:", s, "SAMPLED O:", sampled_o, "NEXT STATE:", s_, "REAL O:", o, "FACTORIDX:", factor_idx, "ACTION:", a, sampled_o != o)
                continue
        return Particles(fp)

    def build_initial_belief(self, num_particles : int) -> Particles:
        # Use the LHS of the simulator if available, else make use of the interface method "sample_initial".
        if hasattr(self.simulator, "sample_multiple_initial"):
            plist = self.simulator.sample_multiple_initial(num_particles)
        else:
            plist = [self.simulator.sample_initial() for _ in range(num_particles)]
        try:
            hash(plist[0])
        except TypeError:
            # Make explicitly unmutable by abusing tuple type.
            plist = self.particle_mapper(tuple, plist)

        if self.weighted:
            w = 1 / len(plist)
            return WPF([(x, w) for x in plist])
        else:
            return Particles(plist)

    def particle_deprivation(self) -> Particles:
        """
        Particle deprivation means we have to rebuild the particle filter from scratch (the initial belief).
        """
        return self.build_initial_belief(self.minimum_n_particles)

    def is_terminal_belief(self, belief : WPF):
        return not any([(not self.simulator.is_final(s)) * (w>0) for s, w in belief])

    def fG_PF(self, belief : WPF, a, factors):
        s_0 = belief.sample()
        _, o, *_ = self.simulator.step(s_0, a)
        new_belief = WPF([])
        rho = total_w = 0
        for s, w in belief:
            # s_, _, r, *_ = self.simulator.step(s, a)
            s_, _, r, terminal = self.particle_step(s, a)
            if terminal: continue
            w_ = w * self.simulator.obs_prob(s_, a, o)
            new_belief.add((s_, w_))
            rho += r * w
            total_w += w
        if len(new_belief) == 0:
            return new_belief, rho, o, True
        else:
            rho /= total_w
        return new_belief, rho, o, False

    def G_PF(self, belief : WPF, a, factored=False):
        s_0 = belief.sample()
        _, o, *_ = self.simulator.step(s_0, a)
        new_belief = WPF([])
        rho = total_w = 0
        for s, w in belief:
            # s_, _, r, *_ = self.simulator.step(s, a)
            s_, _, r, terminal = self.particle_step(s, a)
            if terminal: continue
            if factored:
                w_ = w * self.simulator.obs_prob(s_, a, o) # next state for obs prob
            else:
                w_ = w * self.simulator.obs_prob(s_, a, o) # next state for obs prob
            new_belief.add((s_, w_))
            rho += r * w
            total_w += w
        if self.is_terminal_belief(new_belief):
            return new_belief, rho, o, True
        else:
            rho /= total_w
            new_belief.normalise() # normalization
            # new_belief = WPF([(s_, w_ / total_w) for (s_, w_) in new_belief]) # normalization
        return new_belief, rho, o, False
