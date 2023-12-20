from __future__ import annotations
from functools import partial
import math

import random
import copy
import os
import time

from multiprocessing import Pool, cpu_count

import itertools

from textwrap import dedent
import types
from envs.capturetarget.capturetarget import CaptureTarget
from envs.ma_rock_sample.rs_simulator import build_mars
# from envs.multitaxi import build_multi_taxi

from mpomcp.alpha_uct import ALPHA_POUCT
from mpomcp.particles import ParticleFilterException
from envs.simulator import Simulator
from envs.fff import JointFireFightingSimulator, build_fff
from envs.capturetarget.capturetarget_simulator import build_ct, rebuild_ct

import scipy.stats as st

import numpy as np
import pandas as pd

import json

from tqdm import tqdm
import cProfile, pstats

class EpisodeRunner:

    def __init__(self, algorithm_class : ALPHA_POUCT, sim : Simulator, experiment_id : str, using_pomcp=True, random_policy=False, rand_errors=False, print_debug=False, save_intermediate=False, dynamic_graph=False, render=False,seed=0) -> None:
        self._planner : ALPHA_POUCT = algorithm_class
        if using_pomcp and not random_policy: assert self._planner.use_particles, "We're assuming that the planner uses a particle representation since the UCT parameter is False."
        self.sim : Simulator = sim
        self.using_pomcp = using_pomcp
        self.random_policy = random_policy
        self.rand_errors = rand_errors
        # for debugging purposes
        self.save_intermediate_results = save_intermediate
        self.experiment_id = experiment_id
        self.init_debug()
        self.heavy_debug = print_debug
        self.render = render
        self.seed = seed
        self.dynamic_graph_during_episode = dynamic_graph
    
    def init_debug(self) -> None:
        self.true_states = []
        self.taken_actions = []
        self.trees = []
    
    def update_belief(self) -> None:
        # TODO: for POUCT?
        pass

    def save(self, episode : int, path : str = "./output/episode_runner") -> None:
        """
        Save (debugging) results of experiments to disk. 
        """
        def flatten(arr : np.ndarray, first=True) -> np.ndarray:
            if first: # First axes
                return arr.reshape(-1, *arr.shape[2:])
            else: # Last axes
                return arr.reshape(*arr.shape[:-2], -1)

        def double_save(arr : np.array, name : str) -> None:
            np.savetxt(f"{dir}/{name}.txt", flatten(arr), fmt="%i" if arr.dtype == int else "%s")
            np.save(f"{dir}/{name}.npy", arr, allow_pickle=True)

        dir = f"{path}/{self.experiment_id}/{episode}"
        os.makedirs(dir, exist_ok=True)

        for (l, n) in [(np.array(self.true_states, dtype=int), "true_states"), (np.array(self.taken_actions, dtype=int), "taken_actions"), (np.array(self.trees), "trees")]:
            double_save(l, n)


    def run_episodes(self, horizon, num_episodes, print_progress=True):
        rewards = np.full((num_episodes, horizon), np.nan)
        for i in (tqdm(range(num_episodes)) if print_progress else range(num_episodes)):
            set_random_seeds(self.seed + i)
            rewards[i] = self.run_episode(horizon, episode_debug=(True if num_episodes == 1 else False))
            if self.save_intermediate_results: self.save(i); self.init_debug()
        return rewards
    
    def plan(self, belief, prev_state):
        try:
            action, trees, t_elapsed, n_sims, returns, depths_reached = self._planner.plan(belief, prev_state)
        except ParticleFilterException as e:
            if self.rand_errors:
                print("Planning resulted in Exception! Defaulting to random policy. Error:", e)
                action = self.sim.get_random_action(None)
                self.random_policy = True
                return action, [], 0, 0, [], 0
            else:
                raise e
        # Normal return
        return action, trees, t_elapsed, n_sims, returns, depths_reached
    
    def run_episode(self, horizon, episode_debug = False):
        set_random_seeds(self.seed)
        h = 0
        belief = None if self.using_pomcp or self.random_policy or self._planner.fully_observable else np.ones(self.sim.get_num_states()) / self.sim.get_num_states()
        rewards = np.zeros(horizon)
        prev_state = copy.deepcopy(self.sim.get_state())
        episode_states, episode_actions, episode_trees = [], [], []
        if not self.random_policy: assert self.sim == self._planner.simulator
        total_sims = depths = 0
        while h < horizon:
            if self.render:
                self.sim._simulator.render(prev_state)
                if self.random_policy:
                    time.sleep(0.5)
            if self.random_policy:
                action = self.sim.get_random_action(None)
            else:
                self.sim.set_state(prev_state)
                action, trees, t_elapsed, n_sims, returns, depths_reached = self.plan(belief, prev_state)
                total_sims += n_sims
                depths += np.mean(depths_reached)


            if self.save_intermediate_results:
                episode_states.append(prev_state)
                episode_actions.append(action)

            next_state, o, rewards[h], n_steps = self.sim.step(prev_state, action)
            # rewards[h] -= (self.sim._simulator._num_houses * self.sim._simulator._nf)
            if episode_debug:
                print("#############")
                print("H", h, "S:", prev_state, "S':", next_state, "O:", o, "R:", rewards[h], "A:", action)
                if self.random_policy:
                    trees = []
                else:
                    print("TIME:", t_elapsed, "SIMS:", n_sims)
                print("#############")
            prev_state = copy.deepcopy(next_state)

            if episode_debug and not self.random_policy and self._planner.use_particles:
                pfs = []
                for t in self._planner._factors:
                    try:
                        try:
                            L = t.belief.likelihood
                        except Exception:
                            L = None
                        pfs.append((t.belief.get_entropy(), L, len(t.belief), sorted(t.belief.get_histogram().items(),key=lambda x : x[1], reverse=True)[:5]))
                        # pfs.append(t.belief.mpe())
                    except Exception as e:
                        print("ERROR!", t, e)

                if not self.random_policy: print("PF AFTER SEARCH", *pfs, "GLOBAL:", (self._planner.global_particles_belief.get_entropy(), sorted(self._planner.global_particles_belief.get_histogram().items(), key=lambda x : x[1])[:3]) if self._planner.use_particles and self._planner.global_particles else None, sep='\n')
                if self._planner.factored_filter: print("FPF BEFORE", len(self._planner.factored_filter_belief), self._planner.factored_filter_belief)

            if not self.random_policy:
                if self._planner.FLAT_POMCP: action = [action]
                tik = time.time()
                self._planner.update(action, o)
                tok = time.time()
                if episode_debug:
                    print("Updating particle filters took", tok - tik, "seconds.")
                if self.save_intermediate_results: episode_trees.append(trees)
                if not self.using_pomcp: belief = self.update_belief()
                if self.heavy_debug: print(trees)
            
            if self.dynamic_graph_during_episode:
                assert isinstance(self.sim._simulator, CaptureTarget), "Dynamic Coordination Graphs is currently only implemented for CaptureTarget."
                simu, cg = rebuild_ct(self.sim._simulator, self._planner.graph.num_agents, horizon=(horizon-h), joint=self._planner.FLAT_POMCP, teams=True)
                self._planner.replace_graph(cg)
                self._planner.replace_simulator(simu)
                self.sim = simu

            if episode_debug and not self.random_policy and self._planner.use_particles:
                pfs = []
                for t in self._planner._factors:
                    try:
                        try:
                            L = t.belief.likelihood
                        except Exception:
                            L = None
                        pfs.append((t.belief.get_entropy(), L, len(t.belief), sorted(t.belief.get_histogram().items(),key=lambda x : x[1], reverse=True)[:5]))
                        # pfs.append(t.belief.mpe())
                    except Exception as e:
                        print("ERROR!", t, e)

                if not self.random_policy: print("PF AFTER UPDATE", *pfs, "GLOBAL:", (self._planner.global_particles_belief.get_entropy(), sorted(self._planner.global_particles_belief.get_histogram().items(), key=lambda x : x[1], reverse=True)[:3]) if self._planner.use_particles and self._planner.global_particles else None, sep='\n')
                if self._planner.factored_filter: print("FPF AFTER", len(self._planner.factored_filter_belief), self._planner.factored_filter_belief)

            h += 1
            if n_steps == 0:
                # Reached final state/end of episode
                break
        if self.render:
            self.sim._simulator.render(prev_state)
            if self.random_policy:
                time.sleep(0.5)
        if self.save_intermediate_results:
            self.true_states.append(episode_states)
            self.taken_actions.append(episode_actions)
            self.trees.append(episode_trees)
        return rewards, h, total_sims, depths / h

def f(e : EpisodeRunner, h : int, smosh_errors=False) -> tuple[np.array, int]:
    """
    Helper function for multithreading.
    """
    try:
        return e.run_episode(h)
    except Exception as ve:
        if smosh_errors:
            print(ve)
            return ve
        else:
            raise ve

def g(t : tuple[EpisodeRunner, int], **kwargs) -> tuple[np.array, int]:
    """
    Helper function for multithreading.
    """
    return f(t[0], t[1], **kwargs)

def run_experiment(num_agents, horizon, use_max_plus, joint=False, num_episodes=100, random=False, num_sims=100, no_particles=False, exploration_const=math.sqrt(2), max_time=1, discount=1, factored_statistics=False, save = False, mmdp = False, env : str = "fff", multithreaded = False, dynamic_graph = False, render = False, smosh_errors=False, **kwargs):
    set_random_seeds(kwargs['seed'])
    env = env.lower()
    if env in {"fff", "ffg", "gff"}:
        builder = build_fff
    elif env.startswith("ct") or env.startswith("capturetarget"):
        if '-' in env:
            _, t = env.split('-')
            builder =  partial(build_ct, topology=t)
        else:
            builder = build_ct
    elif env.startswith("mars") or env.startswith("rs") or env.startswith("rocksample"):
        if '-' in env:
            _, n, *k = env.split('-')
            if len(k) > 1:
                k, topology = k
            else:
                topology = 'line'
                k = k[0]
            builder = partial(build_mars, size=int(n), num_rocks=int(k), topology=topology)
        else:
            builder = build_mars
    else:
        ValueError("Unsupported environment as argument.")

    sim, graph = builder(num_agents, horizon, joint, **kwargs)

    print("RENDERING:", render)
    if num_agents < 10:
        print("Edges:", graph.edges)

    ft = not (factored_statistics or joint)
    flat = not factored_statistics and not ft

    assert sum([ft, flat, factored_statistics]) == 1, ("Only one should be true:", [ft, flat, factored_statistics])

    def create_planner(sim) -> ALPHA_POUCT:
        model = None
        return ALPHA_POUCT(
            # Parameters:
            num_sims, discount_factor=discount, use_varel=(not use_max_plus), exploration_const=exploration_const, max_time=max_time, use_particles=(not no_particles),
            simulator=sim, num_factors=len(graph.edges), graph=graph, model=model, fully_observable=mmdp, factored_trees=ft, factored_statistics=factored_statistics, **kwargs)

    if random:
        planner = lambda _ : None
    else:
        planner = create_planner

    # Take at most half the CPU threads.
    if multithreaded is not None:
        if multithreaded > 0: 
            thread_count = multithreaded
        else:
            thread_count = min(num_episodes, cpu_count() // 2)
        multithreaded = True
    else:
        multithreaded = False

    print(dedent(f"""
        --- Running M{"POMCP" if (not no_particles) else "POUCT"} experiment with {num_episodes} episodes on '{env}': 
            => #simulations, max_time, max_depth : {num_sims}, {max_time}, {kwargs['max_depth']}
            => #agents, horizon, #factors        : {num_agents}, {horizon}, {1 if joint else len(graph.edges)}
            => Action selection                  : {"NONE" if flat else "RANDOM" if random else "MP" if use_max_plus else "VE"} ({"N.A." if random else "FLAT POMCP" if flat else "Factored Statistics" if factored_statistics else "Factored Trees"}) {"(PFT)" if kwargs['pft'] else ""}
            => #particles (#mininum), extras     : {kwargs['num_particles']} ({max(20, kwargs['num_particles']//5)}){" +WPF" if kwargs['weighted_particle_filtering'] else ""}{" +PW" if kwargs['progressive_widening'] else ""}{" +LS" if kwargs['likelihood_sampling'] else ""}
            => Exploration const, discount       : {exploration_const}, {discount}
            => Multithreaded (num_threads), seed : {multithreaded} ({thread_count if multithreaded else 1}), {kwargs['seed']}
        """))

    # simus = [builder(num_agents, horizon, joint, **kwargs)[0] for _ in range(num_episodes)]
    simus = [copy.deepcopy(sim) for _ in range(num_episodes)]
    for sim in simus: sim.reset()
    e = [(EpisodeRunner(planner(sim), sim,  using_pomcp=(not no_particles), random_policy=random, rand_errors=kwargs['rand_errors'], save_intermediate=False, experiment_id=kwargs['id'], dynamic_graph=dynamic_graph, render=render, seed=kwargs['seed'] + i), horizon) for i, sim in enumerate(simus)]

    start = time.time()

    if multithreaded:    
        with Pool(thread_count) as p: # might be too superficial to duplicate memory with Pool. 
            # Faster but no intermediate results:
            # result = p.starmap(f, e)
            result = []
            # Slower but with intermediate results -> progress bar:
            with tqdm(total=num_episodes) as pbar:
                for vals in p.imap_unordered(partial(g, smosh_errors=smosh_errors), e):
                    crash = isinstance(vals, Exception)
                    if crash:
                        pass # already handled in g.
                    else:
                        result.append(vals)
                    pbar.update()

    else:
        if num_episodes > 1:
            result = list(tqdm(itertools.starmap(partial(f, smosh_errors=smosh_errors), e), total=num_episodes))
        else:
            result = [e[0][0].run_episode(e[0][1], episode_debug=(True if num_episodes == 1 else False))]

    discounted_rewards = rewards = np.array([x[0] for x in result])
    episode_lengths = np.array([x[1] for x in result])
    number_of_sims = np.array([x[2] for x in result])
    depths = np.array([x[3] for x in result])

    if discount < 1:
        discounted_rewards = np.array([[(discount**step) * r for (step, r) in enumerate(episode)] for episode in rewards])

    print("Experiment took", np.round(time.time() - start, decimals=2), "seconds.")
    print("Average episode length:", episode_lengths.mean(), "/", horizon)
    print("Average number of simulations:", (number_of_sims / episode_lengths).mean(), "/", num_sims, "with max", max_time, "seconds.")
    print("Depth statistics averaged over simulations and horizon:", pd.DataFrame(depths).describe(), sep='\n')

    return discounted_rewards, e, rewards
    
def build_args():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--random", action="store_true", default=False, help="Use random policy, e.g. for baseline result.")
    args.add_argument('--joint', action="store_true", default=False, help="Run experiment using joint action and observation space, as in vanilla POMCP/Sparse-PFT.")
    args.add_argument('--num_agents', '--n', default=4, type=int)
    args.add_argument('--horizon', '--h', default=10, type=int)
    args.add_argument('--action_coordination', choices=['ve', 'mp'], default='ve', type=str)
    args.add_argument('--num_episodes', '--episodes', default=10, type=int, help="Number of episodes to run.")
    args.add_argument('--num_sims', '--sims', default=100, type=int, help="Maximum number of simulation function calls in the tree search.")
    args.add_argument('--max_time', '--time', default=1, type=float, help="Maximum time spent in the tree search in seconds.")
    args.add_argument('--exploration_const', "--c", default=1.25, type=float, help="UCB1 exploration constant c.")
    args.add_argument('--discount', "--gamma", default=1.0, type=float, help="Discount factor in floats (should meet 0 <= y <= 1).")
    args.add_argument('--no_particles', action="store_true", default=False, help="Do not use particle filters. The fallback is to run with POUCT, i.e. with a belief distribution, which might not be implemented for every environment.")
    args.add_argument('--num_particles', '--np', '--p', default=100, type=int, help="Specify the number of particles in each factored filter or in the joint filter, depending on the algorithm set-up.")
    args.add_argument('--max_depth', type=int, default=10, help="Maximum depth of the tree.")
    args.add_argument('--dont_reuse_trees', action="store_true", default=False, help="Rebuild tree every step in the episode, not making use of previous tree search results.")
    args.add_argument('--mmdp', action="store_true", default=False, help="Run in MMDP setting. Meaning: pick the true state of the environment in every simulation call instead of sampling from the belief.")
    args.add_argument('--progressive_widening', '--dpw', action="store_true", default=False, help="Add factored progressive widening to the tree search algorithm to increase depth of the search. Might negatively influence results.")
    args.add_argument('--likelihood_sampling', '--ls', action='store_true', default=False, help="Belief Likelihood-based asymmetric sampling.")
    args.add_argument('--weighted_particle_filtering', '--weighted', '--wpf', action='store_true', default=False, help="Use weighted particle filtering, assumes and requires an explicit observation model.")
    args.add_argument('--factored_statistics', '--fs', action='store_true', default=False, help="Factored statistics / value version of the algorithm. Use with --joint only.")
    args.add_argument('--pft', action='store_true', default=False, help="Use the (factored-trees) Particle Filter Tree algorithm.")
    args.add_argument('--use_sim_particles', action='store_true', default=False, help="Merge the updated belief and simulation particles.")
    args.add_argument('--smosh_errors', action='store_true', default=False, help="Ignore exceptions during multithreading and keep executing the remaining episodes.")
    args.add_argument('--rand_errors', action='store_true', default=False, help="Ignore particle filter exceptions during searching and keep executing the remaining episode with a random policy.")
    args.add_argument('--save', action="store_true", default=False, help="Save intermediate results to disk for debugging. Might not work when running multithreaded.")
    args.add_argument('--multithreaded', "--multi", type=int, nargs='?', const=0, default=None, metavar='PERIOD', help="Run episodes multithreaded, every episode runs in its own process. Maximum number of processes is half the number of CPU threads by default but can be supplied.")    
    args.add_argument('--seed', '--s', type=int, default=1337)
    args.add_argument('--id', type=str, default=time.asctime(time.localtime(time.time())), help="Experiment identifier, determines which directory the results are stored to.")
    args.add_argument('--store_results', action='store_true', default=False, help="Store the benchmark results in a CSV.")
    args.add_argument('--render', action='store_true', default=False)
    args.add_argument('env', type=str),
    args.add_argument('experiment_names', type=str, nargs='*', default=[], help="(Optional) give the function identifier of any experiment to run that is available in this file. E.g. `run_vanilla_pomcp`.")
    return args.parse_args()

def set_random_seeds(seed : int) -> None:
    np.random.seed(seed)
    random.seed(seed)

def process_returns(discounted_returns : np.ndarray[np.ndarray[float]], returns : np.ndarray[np.ndarray[float]], args : dict, filename_suffix : str = "unspecified") -> None:
    if args.discount < 1:
        print(pd.DataFrame(np.column_stack([returns.sum(axis=1), discounted_returns.sum(axis=1)]), columns=['cum. return', 'discounted']).describe())
    else:
        print(pd.DataFrame(discounted_returns.sum(axis=1), columns=[f'cum. disc. return']).describe())
    
    if np.size(discounted_returns.sum(axis=1)) > 1:
        # 95% Confidence interval
        print("95% CI:", st.norm.interval(0.95, loc=discounted_returns.sum(axis=1).mean(), scale = st.sem(discounted_returns.sum(axis=1))))
    
    ret_df = pd.DataFrame(returns)
    disc_ret_df = pd.DataFrame(discounted_returns)
    
    if args.store_results:
        path = f"./experiments/benchmarks/{args.id}/{args.seed}"
        os.makedirs(path, exist_ok=True)
        ret_df.to_csv(f"{path}/vanilla_ret_{filename_suffix}.csv", index=False)
        with open(f'{path}/params_{filename_suffix}.json','w') as paramsfile:
            paramsfile.write(json.dumps(vars(args), indent=4))
        if args.discount < 1:
            disc_ret_df.to_csv(f"{path}/disc_ret_{filename_suffix}.csv", index=False)
    
    return ret_df, disc_ret_df

def run_vanilla_pomcp():
    print("Running Vanilla POMCP benchmark")
    args = build_args()
    args.joint = True
    set_random_seeds(args.seed)
    discounted_returns, runner, rewards = run_experiment(use_max_plus=args.action_coordination=='mp', **vars(args))
    process_returns(discounted_returns, rewards, args, "pomcp")

def run_ve_and_mp():
    print("Running VE and MP experiments (two benchmark runs total).")
    for ac in [False, True]:
        # Run with ac
        args = build_args()
        # Fair state for experiments
        set_random_seeds(args.seed)
        discounted_returns, runner, rewards = run_experiment(use_max_plus=ac, **vars(args))
        process_returns(discounted_returns, rewards, args, f"{'mp' if ac else 've'}")

def run_random():
    print("Running Random Policy")
    args = build_args()
    args.random = True
    set_random_seeds(args.seed)
    discounted_returns, runner, rewards = run_experiment(use_max_plus=args.action_coordination=='mp', **vars(args))
    process_returns(discounted_returns, rewards, args, "random")

def run_all():
    run_random()
    run_ve_and_mp()
    run_vanilla_pomcp()
    print("Done.")

def main(args):
    print("Running default main()")
    print(vars(args))
    discounted_returns, runner, rewards = run_experiment(
        use_max_plus=args.action_coordination=='mp',
        **vars(args))
    process_returns(discounted_returns, rewards, args, f"unknown-{time.time()}")

def profile():
    args = build_args()
    print("Running default main() with cProfile.")
    with cProfile.Profile() as pr:
        main(args)
    stats = pr.create_stats()
    print(stats)
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)

if __name__ in '__main__':
    args = build_args()
    set_random_seeds(args.seed)
    if len(args.experiment_names) == 0:
        main(args)
    else:
        for exp in args.experiment_names:
            print("Running experiment:", exp)
            eval(exp + '()')
