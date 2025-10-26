# Factored Online Planning in Many-Agent POMDPs (AAAI24 Main Track)

All code was ran on an Ubuntu 22.04.3 LTS machine with Python version 3.10.12. The exact requirements with versions can be found in `requirements.txt`. The package dependencies withour explicit versions are in `requirements_loose.txt`, to be used with other (untested) Python3 versions.

## Acknowledgements

- Parts of the code are inspired by the POUCT + particles implementation of [pomdp-py](https://github.com/h2r/pomdp-py), and [this fork](https://github.com/jcsaborio/POMCP) David Silver's POMCP code, as the original is not available anymore.
- Credit is also due for some of the environments. Some of the code of the FFG environment comes from the implementation in [MADP](https://github.com/MADPToolbox/MADP). MARS is a Python variant of this [MARS environment](https://github.com/AdaCompNUS/hyp-despot/tree/master/src/HyP_examples/ma_rock_sample), and CaptureTarget; which was built from the code of [ROLA](https://github.com/yuchen-x/ROLA/blob/main/src/marl_envs/my_env/capture_target.py).

## Requirements

Install basic requirements via `pip install -r requirements.py`.  Ensure `Python3` and the `venv` and `pip` modules are installed, which typically can be installed together with how you have decided to install `python3`.

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -r requirements.txt
```

If installing the exact versions specified in the requirements does not work, please try installing the dependencies without the specific versions (as found in `requirements_loose.txt`). There is a decent chance the code will still work. For instance, we have been able to run the code with Python3 3.12.10 as well.

## Running the code

All experiments can be run with the convenience script.

```
bash run_exp.bash
```

## Output

After running the above commands, or any of the individual runs defined in `run_exp.bash`, the outputs will be in `experiments/benchmarks/$ID/$SEED`, where the seed $SEED is 1337 by default and $ID is set by the `--id` parameter when starting the run. The outputs consists of two `.csv` files per experiment, one with discounted and one with undiscounted rewards, a `params.json` containing parameters, and a large serialization of the results dictionary built during the experiment; named `results.pickle`. This pickle file can be loaded for conveniently postprocessing the results for the paper, for example as we do in the Jupyter notebook file `plot_results_final.ipynb`.

An individual example taken from the `run_exp.bash` is as follows:

```
python3 run_experiments.py fff --max_time 5 --episodes 100 --id fff_exp --multi 34 | tee fff.log
```

This will produce outputs, namely the results pickle, to `experiments/benchmarks/fff_exp/1337/`. In the notebook (`notebooks/plot_results_final.ipynb`), one can then replace the instances of `dirname` with `experiments/benchmarks/fff_exp/1337/results.pickle` appended to your working directory, or e.g., `../experiments/benchmarks/fff_exp/1337/results.pickle` when executing the notebook from the same base folder. The notebook contains separate Markdown headings for producing the results of each experiment.

### Individual Examples

The file `run_experiments.py` is the main starting point for reproducing the experiments in the paper. It controls `episode_runner.py`, which is the main entrypoint to run a certain number of episodes on an environment.

The `run_experiments.py` file can be started with a few parameters, see experiment_helper for the commands used for the paper. Keep in mind this starts 34 threads each time `episode_runner.py` is called by default (since the argument `--multi 34` is passed).

See the `--help` output of `episode_runner.py` to run specific instances. It's copied below for convenience.

```
usage: episode_runner.py [-h] [--random] [--joint] [--num_agents NUM_AGENTS] [--horizon HORIZON] [--action_coordination {ve,mp}] [--num_episodes NUM_EPISODES] [--num_sims NUM_SIMS] [--max_time MAX_TIME] [--exploration_const EXPLORATION_CONST] [--discount DISCOUNT] [--no_particles]
                         [--num_particles NUM_PARTICLES] [--max_depth MAX_DEPTH] [--dont_reuse_trees] [--mmdp] [--progressive_widening] [--likelihood_sampling] [--weighted_particle_filtering] [--factored_statistics] [--pft] [--use_sim_particles] [--smosh_errors] [--rand_errors] [--save]
                         [--multithreaded [PERIOD]] [--seed SEED] [--id ID] [--store_results] [--render]
                         env [experiment_names ...]

positional arguments:
  env
  experiment_names      (Optional) give the function identifier of any experiment to run that is available in this file. E.g. `run_vanilla_pomcp`.

options:
  -h, --help
  show this help message and exit
  --random
  Use random policy, e.g. for baseline result.
  --joint
  Run experiment using joint action and observation space, as in vanilla POMCP/Sparse-PFT.
  --num_agents NUM_AGENTS, --n NUM_AGENTS
  --horizon HORIZON, --h HORIZON
  --action_coordination {ve,mp}
  Denotes the action selection algorithm, variable elimination (VE) or MaxPlus (MP)
  --num_episodes NUM_EPISODES, --episodes NUM_EPISODES
  Number of episodes to run.
  --num_sims NUM_SIMS, --sims NUM_SIMS
  Maximum number of simulation function calls in the tree search.
  --max_time MAX_TIME, --time MAX_TIME
  Maximum time spent in the tree search in seconds.
  --exploration_const EXPLORATION_CONST, --c EXPLORATION_CONST
  UCB1 exploration constant c.
  --discount DISCOUNT, --gamma DISCOUNT
  Discount factor in floats (should meet 0 <= y <= 1).
  --no_particles
  Do not use particle filters. The fallback is to run with POUCT, i.e. with an explicit belief distribution, which might not be implemented or feasible for every environment.
  --num_particles NUM_PARTICLES, --np NUM_PARTICLES, --p NUM_PARTICLES
  Specify the number of particles in each factored filter or in the joint filter, depending on the algorithm set-up.
  --max_depth MAX_DEPTH
  Maximum depth of the search tree(s).
  --dont_reuse_trees
  Rebuild tree every step in the episode, not making use of previous tree search results.
  --mmdp
  Run in MMDP setting. Meaning: pick the true state of the environment in every simulation call instead of sampling from the belief.
  --progressive_widening, --dpw
  Add factored progressive widening to the tree search algorithm to increase depth of the search. Might negatively influence results.
  --likelihood_sampling, --ls
  Belief Likelihood-based asymmetric sampling.
  --weighted_particle_filtering, --weighted, --wpf
  Use weighted particle filtering, assumes and requires an explicit observation model.
  --factored_statistics, --fs
  Factored statistics / value version of the algorithm. Use with --joint only.
  --pft
  Use the (factored-trees) Particle Filter Tree algorithm.
  --use_sim_particles
  Merge the updated belief and simulation particles.
  --smosh_errors
  Ignore exceptions during multithreading and keep executing the remaining episodes.
  --rand_errors
  Ignore particle filter exceptions during searching and keep executing the remaining episode with a random policy.
  --save
  Save intermediate results to disk for debugging. Might not work when running multithreaded.
  --multithreaded [PERIOD], --multi [PERIOD]
  Run episodes multithreaded, every episode runs in its own process. Maximum number of processes is half the number of CPU threads by default but can be supplied.
  --seed SEED, --s SEED
  --id ID
  Experiment identifier, determines which directory the results are stored to.
  --store_results
  Store the benchmark results in a CSV.
  --render
```
