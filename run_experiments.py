from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
import os

import pandas as pd
import numpy as np
import json
import pickle
#from envs.sysadmin import ring_of_ring_edges

from episode_runner import run_experiment, process_returns, set_random_seeds, build_args

def update_arguments(args, n_agent, h, ac, random) -> dict:
    args.num_agents = n_agent
    args.horizon = h
    args.use_max_plus = ac == 'mp'
    args.random = random
    return args

def update_arguments2(args, n_agent, h, ac, random) -> dict:
    args['num_agents'] = n_agent
    args['horizon'] = h
    args['use_max_plus'] = ac == 'mp'
    args['random'] = random
    return args

def run_experiment_set(args : Namespace,
                       cs : list[int],
                       n_agents : list[int], 
                       horizon : list[int], 
                       sims : list[int], 
                       discount : float, 
                       max_time : int, 
                       num_particles,
                       max_depth : int,
                       fpfs = [False],
                       action_selection = ['ve', 'mp'], # MP or VE
                       algorithms = ['pomcp', 'fs-pomcp', 'ft-pomcp', 'pft', 'fs-pft', 'ft-pft'],
                       msts = [False, True], # MST or no
                       wpf = [True, False],
                       number_of_joint_particles = lambda n_agent, n_particles : (n_agent-1) * n_particles,
                       particle_exp = False,
                       ):
    
    result_path = f"./experiments/benchmarks/{args.id}/{args.seed}/results.pickle"
    if os.path.exists(result_path):
        print("Result output path already exists, updating existing dictionary with missing runs.")
        with open(result_path, 'rb') as handle:
            results_dict = pickle.load(handle)
    else:
        print("Creating new results to:", result_path)
        results_dict = {}
    print("ALGORITHMS: ", algorithms, "N_AGENTS:", n_agents)
    args.discount = discount
    args.max_time = max_time
    args.max_depth = max_depth
    args.num_particles = num_particles
    args.store_results = True
    args.rand_errors = args.smosh_errors = True
    for n_agent in sorted(n_agents):
        if not n_agent in results_dict:
            results_dict[n_agent] = {}
        print("-"*1, "agents:", n_agent, )
        for h in horizon:
            print("-"*2, "h:", n_agent)
            # RANDOM
            arguments = update_arguments(args, n_agent, h, None, True)
            prev = arguments.num_episodes
            arguments.num_episodes = 10000
            discounted_returns, _, rewards = run_experiment(**vars(arguments))
            arguments.num_episodes = prev
            name = f"{args.env}_n{n_agent}_h{h}_RANDOM"
            ret_df, disc_ret_df = process_returns(discounted_returns, rewards, arguments, name)
            results_dict[n_agent][name] = {}
            results_dict[n_agent][name]['discounted'] = disc_ret_df
            results_dict[n_agent][name]['undiscounted'] = ret_df
            if n_agent > 65:
                # Allow more memory per thread.
                if args.multithreaded > 8: args.multithreaded = 8
            for c in cs:
                args.exploration_const = c
                print("-"*3, "c:", c)
                for s in sims:
                    if not s in results_dict[n_agent]:
                        results_dict[n_agent][num_particles if particle_exp else s] = {}
                    args.num_sims = s
                    print("-"*4, "sims:", s)
                    for algo in algorithms:
                        print("-"*5, "algorithm:", algo)
                        for filtering in wpf:
                            print("-"*6, "wpf:", filtering)
                            args.weighted_particle_filtering = filtering
                            for fpf in fpfs:
                                args.factored_pf = fpf
                                print("-"*7, "fpf:", fpf)
                                if fpf and not filtering:
                                    print("Not runing FPF without weighted filtering.")
                                    continue
                                args.use_sim_particles = not filtering
                                # args.joint = algo in ['pomcp', 'pomcpow', 'fs-pomcpow' 'fs-pomcp', 'pft', 'fs-pft']
                                args.joint = not (algo.startswith("ft-"))
                                args.likelihood_sampling = (not args.joint) and filtering and (not fpf)
                                if not ('ft' in algo) and not ('fs' in algo) and fpf:
                                    print("Not running flat POMCP/PFT algorithms with FPF.")
                                    continue
                                if args.joint: 
                                    args.num_particles = number_of_joint_particles(n_agent, num_particles)
                                else:
                                    args.num_particles = num_particles
                                args.factored_statistics = 'fs' in algo
                                args.pft = 'pft' in algo
                                args.pomcpow = 'pomcpow' in algo
                                if (args.pft or args.pomcpow) and not filtering:
                                    print("Not running POMCPOW/PFT algorithm without WPF.")
                                    continue # Don't run PFT with unweighted particle filtering
                                if args.joint and not args.factored_statistics:
                                    print("-"*8, "ac: flat UCB")
                                    arguments = update_arguments(args, n_agent, h, None, False)
                                    name = f"{args.env}_n{n_agent}_h{h}_c{c}_s{s}_{algo}{'_fpf' if fpf else ''}{'_wpf' if filtering else ''}"
                                    if name in results_dict[n_agent][num_particles if particle_exp else s]:
                                        print("Experiment", name, "already exists. Skipping...")
                                        continue
                                    else:
                                        print(name, "does not exist in dictionary:", results_dict[n_agent][num_particles if particle_exp else s].keys())
                                    if n_agent > 20:
                                        print("Skipping Flat POMCP/PFT with more than 20 agents because of potential memory issues..")
                                        discounted_returns = rewards = np.array([[]])
                                    else:
                                        print("Running:", name)
                                        discounted_returns, _, rewards = run_experiment(**vars(arguments))
                                    ret_df, disc_ret_df = process_returns(discounted_returns, rewards, arguments, name)
                                    results_dict[n_agent][num_particles if particle_exp else s][name] = {}
                                    results_dict[n_agent][num_particles if particle_exp else s][name]['discounted'] = disc_ret_df
                                    results_dict[n_agent][num_particles if particle_exp else s][name]['undiscounted'] = ret_df
                                    with open(result_path, 'wb') as handle:
                                        pickle.dump(results_dict, handle)
                                else:
                                    for ac in action_selection:
                                        print("-"*8, "ac:", ac)
                                        for mst in msts:
                                            print("-"*9, "spanning tree:", mst)
                                            args.spanning_tree = mst
                                            arguments = update_arguments(args, n_agent, h, ac, False)
                                            name = f"{args.env}_n{n_agent}_h{h}_c{c}_s{s}_{algo}_{ac}{'_fpf' if fpf else ''}{'_wpf' if filtering else ''}{'_mst' if mst else ''}"
                                            if name in results_dict[n_agent][num_particles if particle_exp else s]:
                                                print(
                                                    "Experiment", name, "already exists. Skipping..."
                                                )
                                                continue
                                            else:
                                                print(name, "does not exist in dictionary:", results_dict[n_agent][num_particles if particle_exp else s].keys())
                                            if n_agent > 65 and ac == 've' and not mst:
                                                print("Not running non-MST VE on problems with more than 60 agent due to memory issues.")
                                                discounted_returns = rewards = np.array([[]])
                                            else:
                                                print("Running:", name)
                                                print(vars(arguments))
                                                discounted_returns, _, rewards = run_experiment(**vars(arguments))
                                            ret_df, disc_ret_df = process_returns(discounted_returns, rewards, arguments, name)
                                            results_dict[n_agent][num_particles if particle_exp else s][name] = {}
                                            results_dict[n_agent][num_particles if particle_exp else s][name]['discounted'] = disc_ret_df
                                            results_dict[n_agent][num_particles if particle_exp else s][name]['undiscounted'] = ret_df
                                            with open(result_path, 'wb') as handle:
                                                pickle.dump(results_dict, handle)
        
        # with open(f"./experiments/benchmarks/{args.id}/{args.seed}/results.json", 'w') as result_file:
            # result_file.write(json.dumps(results_dict, indent=4))
    
def run_fff(args):
    run_experiment_set(args, n_agents=[4, 16, 32, 64], sims=[1000], cs=[5], horizon=[10], discount=0.99, max_depth=10, num_particles=20, max_time = args.max_time, msts = [False])

def run_sysadmin_ring(args):
    run_experiment_set(args, 
                       n_agents=[4, 16, 32, 64], 
                       sims=[10, 25, 50, 100, 250, 1000], 
                       cs=[5], horizon=[10], discount=0.95, max_depth = 25, 
                       num_particles=20, max_time = 5,
                       msts = [False, True], wpf=[False], algorithms = ['pomcp', 'fs-pomcp', 'ft-pomcp'],
                       number_of_joint_particles = lambda n_agent, n_particles : n_agent * n_particles)

def run_sysadmin_star(args):
    run_experiment_set(args, 
                       n_agents=[4, 16, 32, 64], 
                       sims=[10, 25, 50, 100, 250, 1000], 
                       cs=[5], horizon=[10], discount=0.95, max_depth = 25, 
                       num_particles=20, max_time = 5,
                       msts = [False], wpf=[False], algorithms = ['pomcp', 'fs-pomcp', 'ft-pomcp'])

def run_sysadmin_ring_of_ring(args):
    run_experiment_set(args,
                       n_agents=[6, 18, 36, 72], 
                       sims=[10, 25, 50, 100, 250, 1000], 
                       cs=[5], horizon=[50], discount=0.95, max_depth=25,
                       num_particles=20, 
                       max_time = 5, 
                       msts = [False, True], 
                       wpf=[False],# True], 
                       algorithms = ['pomcp', 'fs-pomcp', 'ft-pomcp'], 
                       number_of_joint_particles = lambda n_agent, n_particles : len(ring_of_ring_edges(n_agent)) * n_particles)

def get_lambda(topology : str):
        if topology == 'line':
            return lambda n_agent, n_particles : (n_agent-1) * n_particles
        elif topology == 'team':
            return lambda n_agent, n_particles : (n_agent // 2) * n_particles
        else:
            raise ValueError()

def run_mars(args):
    # agents = [[[3], 'line'], [[4], 'team'], [[4], 'line'], [[5], 'line'], [[6], 'team'], [[6], 'line']]
    agents = [[[3], 'line'], [[4], 'team'], [[5], 'line'], [[6], 'team']]
    problems = reversed([(7, 8), (11, 11), (15, 15)])
    # problems = reversed([(15, 15)])
    args.max_depth = 25
    set_of_algorithms = ['pomcp', 'fs-pomcp', 'ft-pomcp', 'pft', 'fs-pft', 'ft-pft']
    # set_of_algorithms = ['pomcp', 'pomcpow', 'fs-pomcpow', 'fs-pomcp', 'ft-pomcp', 'pft', 'fs-pft', 'ft-pft']
    for problem_size in problems:    
        size, n_rocks = problem_size
        for n_agents, topology in agents:
            args.env = f'mars-{size}-{n_rocks}-{topology}'
            algs = [alg for alg in set_of_algorithms if (n_agents[0] > 5 and (alg.startswith('ft-') or alg.startswith('fs-'))) or n_agents[0] <= 5]
            run_experiment_set(args, n_agents=n_agents, algorithms=algs, cs=[1.25], fpfs=[False], horizon=[40], discount=0.95, max_depth=25, sims=[10_000], max_time=args.max_time, num_particles=100, msts = [False], action_selection=['ve'], number_of_joint_particles=get_lambda(topology))

def run_ct(args,):
    agents = [[3], [4], [5], [6]]
    topologies = ['line', 'team', 'line', 'team']
    set_of_algorithms = ['pomcp', 'fs-pomcp', 'ft-pomcp']
    set_of_algorithms += ['pft', 'fs-pft', 'ft-pft']
    for n_agents, topology in zip(agents, topologies):
        env = f'ct-{topology}'
        print(env)
        args.env = env
        run_experiment_set(args, n_agents=n_agents,algorithms=set_of_algorithms, horizon=[50], cs=[0.5], discount=0.95, max_depth = 10, sims=[10_000], max_time=args.max_time, num_particles=100, msts=[False], action_selection=['ve'], number_of_joint_particles=get_lambda(topology))
        # run_experiment_set(args, n_agents=n_agents,algorithms=set_of_algorithms, horizon=[50], cs=[0.5], discount=0.95, max_depth = 10, sims=[10_000], max_time=15, num_particles=100, msts=[False], number_of_joint_particles=get_lambda(topology))

def run_pf_experiment(args):
    agents = [4, 8, 16]
    consistent_id = str(args.id)
    num_particles = [10, 25, 50, 100, 250, 500, 1000, 2000]
    for num_p in num_particles:
        # args.id = f"{consistent_id}_p{num_p}"
        # print(args.id)
        run_experiment_set(args, n_agents=agents, horizon=[10], cs=[5], fpfs=[True, False], discount=0.99, max_depth = 25, sims=[250], max_time=1, num_particles=num_p, msts=[False])

def main():
    args = build_args()
    if args.env == 'pf':
        args.env = "fff"
        run_pf_experiment(args)
    if args.env == "fff":
        run_fff(args)
    elif args.env == 'sysadmin-ring':
        run_sysadmin_ring(args)
    elif args.env == 'sysadmin-star':
        run_sysadmin_star(args)
    elif args.env == 'sysadmin-ror':
        run_sysadmin_ring_of_ring(args)
    elif args.env == 'mars':
        run_mars(args)
    elif args.env == 'ct':
        run_ct(args)
    else:
        raise ValueError("Invalid experiment env, choose from:", ["fff", 'sysadmin-ring', 'sysadmin-star', 'sysadmin-ror', 'mars',  'ct'])

if __name__ in '__main__':
    main()
