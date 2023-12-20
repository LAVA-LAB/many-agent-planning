# FROM https://github.com/yuchen-x/ROLA.
from functools import reduce
import random
from typing import Union
import numpy as np
import IPython
import gym

from scipy.spatial.distance import cdist

import pprint

from numpy.random import randint

"""
Adapted from base env + render code as in https://github.com/yuchen-x/ROLA.
"""

NORTH = np.array([0, 1])
SOUTH = np.array([0, -1])
WEST = np.array([-1, 0])
EAST = np.array([1, 0])
STAY = np.array([0, 0])

TRANSLATION_TABLE = [
    [WEST, NORTH, EAST],
    [EAST, SOUTH, WEST],
    [SOUTH, WEST, NORTH],
    [NORTH, EAST, SOUTH],
    [STAY, STAY, STAY],
]

DIRECTION = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

# ACTIONS = ["NORTH", "SOUTH", "WEST", "EAST", "STAY"]
ACTIONS = ["NORTH", "SOUTH", "WEST", "EAST"]
AGENT_DIRECTION = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])

class CaptureTarget(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self,
        n_target=1,
        n_agent=2,
        grid_dim=(16, 16),
        terminate_step=60,
        intermediate_r=False,
        target_flick_prob=0.3,
        obs_one_hot=False,
        tgt_avoid_agent=True,
        tgt_random_move=False,
        tgt_trans_noise=0.0,
        agent_trans_noise=0.0,
        restricted_movement=False,
        only_target_obs=True,
    ):

        # env generic settings
        self.n_target = n_target
        self.n_agent = n_agent
        self.multi_task = self.n_target != 1
        self.intermediate_reward = intermediate_r
        self.terminate_step = terminate_step
        self.viewer = None
        self.do_not_normalize = True # !!!
        self.restricted_movement = restricted_movement # !!!
        # dimensions
        self.x_len, self.y_len = grid_dim
        self.x_mean, self.y_mean = np.mean(np.arange(self.x_len)), np.mean(
            np.arange(self.y_len)
        )
        # probabilities
        self.target_flick_prob = target_flick_prob
        self.tgt_avoid_agent = tgt_avoid_agent
        self.tgt_random_move = tgt_random_move
        self.tgt_trans_noise = tgt_trans_noise
        self.agent_trans_noise = agent_trans_noise
        self.sensor_failure = 0.1
        # action space
        self.n_action = [len(ACTIONS)] * n_agent
        self.action_space = [gym.spaces.Discrete(len(ACTIONS))] * n_agent
        # observation mode
        self.obs_one_hot = obs_one_hot
        self.only_target_obs = only_target_obs
        if self.obs_one_hot:
            self.obs_size = [self.x_len * self.y_len] * n_agent # agent position and target position
        else:
            self.obs_size = [len(grid_dim) * 2] * n_agent # agent position and target position
        self.observation_space = [
            gym.spaces.Box(low=-255, high=255, shape=(self.obs_size[0],), dtype=np.float32)
        ] * n_agent
        if not self.n_target == 1 and self.n_agent > 1:
            import warnings
            warnings.warn("Sure you want to run 1 agent?")
        # print("CaptureTarget Variables:")
        # pprint.pprint(self.__dict__)
        self.reset()
        self.max_total_manhattan_distance = self.manhattan([0,0], [self.x_len, self.y_len]) * self.n_agent

    def action_space_sample(self, idx):
        return np.random.randint(self.n_action[idx])

    def action_space_batch_sample(self):
        return np.random.randint(self.n_action[0], size=self.n_agent)

    def reset(self, debug=False):
        """
        Parameter
        ---------
        debug: bool
            if debug, will render the environment for visualization

        Return
        ------
        List[numpy.array]
            a list of agents' observations
        """
 
        self.step_n = 0
        self.visited = np.zeros(self.n_target)
        # "game state" is really just the positions of all players and targets
        self.target_positions = np.stack([self.rand_position() for _ in range(self.n_target)])
        self.agents = [self.rand_position() for _ in range(self.n_agent)]
        self.agent_positions = self.wrap_positions(np.stack(self.agents))
        assert self.target_positions.shape == (self.n_target, 2)
        obs = self.get_obs(debug)
        if debug:
            self.render()
        return obs

    def step(self, actions, debug=False, episode_step=False):
        self.step_n += 1
        assert len(actions) == self.n_agent, f"Length of actions ({len(actions)}) does not match number of agents ({self.n_agent}). Actions: {actions}"

        self.agent_positions = self.move(
            self.agent_positions, actions, noise=self.agent_trans_noise
        )

        if self.tgt_random_move:
            target_directions = np.random.randint(len(TRANSLATION_TABLE), size=self.n_target)
        elif self.tgt_avoid_agent:
            target_directions = self.get_tgt_moves()
        else:
            target_directions = [4] * self.n_target

        self.target_positions = self.move(
            self.target_positions, target_directions, noise=self.tgt_trans_noise
        )

        won = self.target_captured()

        r = float(won) if episode_step else self.reward()

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            print("Target  \t action \t\t{}".format(ACTIONS[target_directions[0]]))
            print(" ")
            print("Agent_0 \t action \t\t{}".format(ACTIONS[actions[0]]))
            print(" ")
            print("Agent_1 \t action \t\t{}".format(ACTIONS[actions[1]]))

        terminate = bool(won) # or self.step_n > self.terminate_step

        return self.get_obs(debug), r, terminate, {}

    def get_obs(self, debug=False):
        return self._get_obs(self.agent_positions, self.target_positions, debug=debug)

    def _get_obs(self, agent_positions, target_positions, debug=False):
        """
        Return
        ------
        List[numpy.arry]
            a list of agents' observations
        """

        if self.obs_one_hot:
            agt_pos_obs = self.one_hot_positions(agent_positions)
            tgt_pos_obs = self.one_hot_positions(target_positions)
        else:
            agt_pos_obs = self.normalize_positions(agent_positions)
            tgt_pos_obs = self.normalize_positions(target_positions)

            if self.n_agent > 1 and self.n_target == 1:
                tgt_pos_obs = np.tile(tgt_pos_obs, (self.n_agent, 1))

        if self.target_flick_prob > 0:
            tgt_pos_obs = self.flick(tgt_pos_obs, prob=self.target_flick_prob)

        if debug:
            print("")
            print("Observations list:")
            for i in range(self.n_agent):
                print(
                    "Agent_" + str(i) + " \t self_loc  \t\t{}".format(agent_positions[i])
                )
                print(
                    "          "
                    + " \t tgt_loc  \t\t{}".format(
                        target_positions[0]
                        if not all(tgt_pos_obs[i] == -1.0)
                        else np.array([-1, -1])
                    )
                )
                print("")

        if self.only_target_obs:
            # return [tuple(obs) for obs in tgt_pos_obs]
            obs = [self.agent_sees_target(agent_pos, target_positions)  for agent_pos in agent_positions]
            return [o if random.random() > self.sensor_failure else 1-o for o in obs]
        else:
            return [tuple(obs) for obs in np.concatenate([agt_pos_obs, tgt_pos_obs], axis=1)]
    
    def agent_sees_target(self, agent_pos : np.ndarray, target_pos : np.ndarray):
        "Agent can see only manhattan style"
        assert target_pos.size == 2, "Only single target currently."
        target_pos = target_pos.ravel()
        return int(agent_pos[0] == target_pos[0] or agent_pos[1] == target_pos[1])

    @property
    def state(self):
        return self.get_state()

    def get_state(self):
        """
        Request environmental state

        Return
        ------
        numpy.array
            the positions of all entities
        """

        agt_pos = self.normalize_positions(self.agent_positions)
        tgt_pos = self.normalize_positions(self.target_positions)
        return np.vstack([agt_pos, tgt_pos]).reshape(-1).copy()

    def set_state(self, state_vector : Union[np.ndarray, tuple]) -> None:
        state_vector = np.array(state_vector)
        tgt_pos = state_vector[-2 * self.n_target:].reshape(self.n_target, 2)
        agt_pos = state_vector[:-2 * self.n_target].reshape(self.n_agent, 2)
        self.agent_positions = agt_pos
        self.target_positions = tgt_pos

    def get_env_info(self):
        return {
            "state_shape": len(self.get_state()),
            "obs_shape": self.obs_size[0],
            "n_actions": self.n_action[0],
            "n_agents": self.n_agent,
            "episode_limit": self.terminate_step,
        }

    def get_avail_actions(self):
        return [[1] * n for n in self.n_action]

    #################################################################################################
    # Helper functions
    # def get_distances_to_target(self, agent_idx):

    def get_heuristic_move(self, agent_idx):
        if self.n_target > 1:
            raise NotImplementedError()
        moves = self.wrap_positions(AGENT_DIRECTION + self.agent_positions[agent_idx])
        h = self.distance(self.target_positions, moves)
        return h.argmin()
        # possibilities = np.where(h == h.min())[0]
        # return random.choice(possibilities)
        return np.random.choice(possibilities, size=1)

    def get_tgt_moves(self, single=False):
        assert self.target_positions.shape[0] == 1
        moves = self.wrap_positions(DIRECTION + self.target_positions)
        cl_agt_idx = self.distance(
            self.agent_positions, self.target_positions
        ).argmin()
        h = 10 * self.distance(self.agent_positions[cl_agt_idx][None, :], moves)
        if not single:
            for i in range(self.n_agent):
                if i != cl_agt_idx:
                    h += self.distance(self.agent_positions[i][None, :], moves)
        return [h.argmax()]

    def move(self, positions, directions, noise=0.0):
        """
        Move each agent

        Parameters
        ----------
        positions: numpy.array
            agents' positions
        directions: list[int]
            a list of agents' moving directions
        noise: float
            noisy transition

        Return
        ------
        numpy.array
            agents' new positions
        """
        translations = np.stack([self.translation(d, noise=noise) for d in directions])
        positions += translations
        return self.wrap_positions(positions)

    def target_captured(self, hard_mode = False):
        positions = [np.array_equal(a, t) for a, t in zip(self.agent_positions, self.respective_target_positions)]
        if hard_mode:
            return all(positions)
        else:
            return any(positions)

    @staticmethod
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def distance(a, b, simple=True):
        if simple:
            return cdist(a, b)
        else:
            return np.linalg.norm(a - b, axis=1)
    
    def reward(self):
        if self.target_captured():
            return 1
        else:
            return 0
        dist = 0
        for agent_idx in range(self.n_agent):
            dist += self.manhattan(self.target_positions[0], self.agent_positions[agent_idx])
        dist /= self.max_total_manhattan_distance
        assert dist > 0 and dist < 1
        return dist

    
    def reward_of_state(self, state):
        state_vector = np.array(state)
        tgt_pos = state_vector[-2 * self.n_target:].reshape(self.n_target, 2)
        agt_pos = state_vector[:-2 * self.n_target].reshape(self.n_agent, 2)
        return self.reward(agt_pos, np.tile(tgt_pos,(self.n_agent,1)))

    def reward_old(self, agent_positions, respective_target_positions, hard_mode = False):
        positions = [np.array_equal(a, t) for a, t in zip(self.agent_positions, self.respective_target_positions)]
        if self.restricted_movement:
            num_captured = sum(positions)
            if num_captured > 2:
                print("HUH?")
                exit()
            elif num_captured == 2:
                return True
            else:
                return False
        else:
            if hard_mode:
                return all(positions)
            else:
                return any(positions)

    @property
    def respective_target_positions(self):
        if self.multi_task:
            return self.target_positions
        else:
            return (self.target_positions[0] for _ in range(self.n_agent))

    def rand_position(self):
        return np.array([randint(self.x_len), randint(self.y_len)])

    @staticmethod
    def translation(direction, noise=0.1):
        if noise > 0:
            return TRANSLATION_TABLE[direction][
                random.choices([0,1,2], weights=[noise / 2, 1 - noise, noise / 2], k=1)[0]
                # np.random.choice(3, p=[noise / 2, 1 - noise, noise / 2])
            ]
        else:
            return TRANSLATION_TABLE[direction][1]

    def flick(self, N, prob=0.3):
        mask = np.random.random(N.shape[0]).reshape(N.shape[0], -1) > prob
        if self.obs_one_hot:
            return N * mask
        else:
            flicker = np.full((N.shape[0], 2), -1)
            # flicker = np.stack([np.array([-1, -1]) for _ in range(N.shape[0])])
            # NOISES:
            # flicker = np.random.randint(-self.y_len // 2, self.y_len // 2, size=(N.shape[0]))
            # flicker = (np.random.normal(size=(N.shape[0])) * self.y_len // 2)
        # return ((N + (flicker * mask)) % self.y_len).astype(int) # Noisy observations
        return N * mask + flicker * np.logical_not(mask) # Original flickering

    def normalize_positions(self, positions):
        if self.do_not_normalize:
            return positions
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([X / self.x_mean, Y / self.y_mean], axis=1)

    def one_hot_positions(self, positions):
        one_hot_vector = np.zeros((self.n_agent, self.x_len * self.y_len))
        index = positions[:, 1] * self.y_len + positions[:, 0]
        one_hot_vector[np.arange(self.n_agent), index] = 1
        return one_hot_vector

    def wrap_positions(self, positions):
        if not self.restricted_movement:
            assert self.x_len == self.y_len
            return np.clip(positions, 0, self.x_len-1) # Wall movement
            return positions % self.x_len # Toroidal movement
        # fix translations which moved positions out of bound.
        X, Y = np.split(positions, 2, axis=1)
    
        if not self.restricted_movement:
            return np.concatenate([np.clip(X, 0, self.x_len-1), np.clip(Y, 0, self.y_len-1)], axis=1)

        if self.restricted_movement and len(X) == len(Y) == self.n_agent:
            # clippers = [((0, 1), (0, 3)), ((0, 3), (0, 1)), ((2, 3), (0, 3)), ((0, 3), (2, 3))]
            clippers = [((0,3), (0,1)), ((0,1), (0,3)), ((0,3), (2,3)), ((2,3), (0,3))]
            for i, (x, y, clips) in enumerate(zip(X, Y, clippers)):
                X[i] = np.clip(x, clips[0][0], clips[0][1])
                Y[i] = np.clip(y, clips[1][0], clips[1][1])
        return np.concatenate([X % self.x_len, Y % self.y_len], axis=1)

    def obs_prob(self, state, act, obs):
        return reduce(lambda x, y : x*y, [self.individual_obs_prob(state, i, a, o, act) for i, (a, o) in enumerate(zip(act, obs))], 1)

    def individual_obs_prob(self, state, agent_i, agent_a, agent_o, joint_a):
        """
        Individual observation prob of a single agent, given the state, agent idx, agent action, and agent observation.
        """
        assert self.n_target == 1
        self.set_state(state)

        agent_o = np.array(agent_o)

        # real_obs, *_ = self.step(joint_a)

        # next_agent_positions = self.move(
        #     self.agent_positions, joint_a, noise=self.agent_trans_noise
        # )

        # if self.tgt_random_move:
        #     target_directions = np.random.randint(len(TRANSLATION_TABLE), size=self.n_target)
        # elif self.tgt_avoid_agent:
        #     target_directions = self.get_tgt_moves()
        # else:
        #     target_directions = [4] * self.n_target

        # next_target_positions = self.move(
        #     self.target_positions, target_directions, noise=self.tgt_trans_noise
        # )

        if self.only_target_obs:
            target_seen = self.agent_sees_target(self.agent_positions[agent_i], self.target_positions)

            assert target_seen in {0, 1}
            target_seen = bool(target_seen)

            if target_seen:
                if agent_o == 1:
                    p=(self.target_flick_prob) * (1-self.sensor_failure) + (1-self.target_flick_prob) * (self.sensor_failure)
                else:
                    # p=1 - ((self.target_flick_prob) * (1-self.sensor_failure) + (1-self.target_flick_prob) * (self.sensor_failure))
                    p = (self.target_flick_prob * self.sensor_failure) + (1-self.target_flick_prob) * (1-self.sensor_failure)
            else:
                if agent_o == 1:
                    p=self.sensor_failure
                    # p*=(1-self.sensor_failure)
                else:
                    p=(1-self.sensor_failure)
            return p
        else:
            raise NotImplementedError()

    def individual_obs_prob_old(self, state, agent_i, agent_a, agent_o, joint_a):
        """
        Individual observation prob of a single agent, given the state, agent idx, agent action, and agent observation.
        """
        assert self.n_target == 1
        self.set_state(state)

        agent_o = np.array(agent_o)

        # real_obs, *_ = self.step(joint_a)

        next_agent_positions = self.move(
            self.agent_positions, joint_a, noise=self.agent_trans_noise
        )

        if self.tgt_random_move:
            target_directions = np.random.randint(len(TRANSLATION_TABLE), size=self.n_target)
        elif self.tgt_avoid_agent:
            target_directions = self.get_tgt_moves()
        else:
            target_directions = [4] * self.n_target

        next_target_positions = self.move(
            self.target_positions, target_directions, noise=self.tgt_trans_noise
        )

        if self.only_target_obs:
            target_seen = self.agent_sees_target(next_agent_positions[agent_i], next_target_positions)

            if target_seen == 1:
                if agent_o == 1:
                    p=(self.target_flick_prob) * (1-self.sensor_failure) + (1-self.target_flick_prob) * (self.sensor_failure)
                else:
                    # p=1 - ((self.target_flick_prob) * (1-self.sensor_failure) + (1-self.target_flick_prob) * (self.sensor_failure))
                    p = (self.target_flick_prob * self.sensor_failure) + (1-self.target_flick_prob) * (1-self.sensor_failure)
            elif target_seen == 0:
                if agent_o == 1:
                    p=self.sensor_failure
                    # p*=(1-self.sensor_failure)
                else:
                    p=(1-self.sensor_failure)
            else:
                raise ValueError()
            return p

        # actual_obs = np.array(self._get_obs(next_agent_positions, next_target_positions))
        # actual_ob = actual_obs[agent_i]
        # if self.only_target_obs:
        #     actual_obs_agent_loc, actual_obs_tgt_loc = None, actual_ob
        # else:
        #     actual_obs_agent_loc, actual_obs_tgt_loc = np.split(actual_ob, 2)

        # obs_agent_loc, obs_tgt_loc = np.split(agent_o, 2)

        # p = 1.0
        # slight_margin = 0 # 1e-5
        # if self.tgt_trans_noise>0 or self.agent_trans_noise>0:
        #     raise NotImplementedError()
        # else:
        #     if obs_tgt_loc.sum() == -2:
        #         return p * self.target_flick_prob
        #     else:
        #         if np.array_equal(obs_tgt_loc, actual_obs_tgt_loc):
        #             if actual_obs_agent_loc is not None:
        #                 if np.array_equal(obs_agent_loc, actual_obs_agent_loc):
        #                     return p * (1-slight_margin) * (1-self.target_flick_prob)
        #             else:
        #                 return p * (1-slight_margin) * (1-self.target_flick_prob)
        #         return slight_margin * (1-self.target_flick_prob)
        

    def render(self, state=None, mode="human"):
        if state is not None:
            self.set_state(state)

        screen_width = 8 * 100
        screen_height = 8 * 100

        scale = 8 / self.y_len

        agent_size = 40.0
        agent_in_size = 35.0
        agent_clrs = [
            ((0.15, 0.15, 0.65), (0.0, 0.4, 0.8)),
            ((0.15, 0.65, 0.15), (0.0, 0.8, 0.4)),
            ((0.65, 0.15, 0.15), (0.8, 0.4, 0.0)),
            ((0.65, 0.65, 0.65), (0.8, 0.8, 0.8)),
        ]

        target_l = 80.0
        target_w = 26.0
        target_in_l = 70.0
        target_in_w = 16.0
        target_clrs = ((0.65, 0.15, 0.15), (1.0, 0.5, 0.5))

        if self.viewer is None:
            from envs.capturetarget import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # -------------------draw agents
            self.render_agents = []
            # agent_clrs = [(0.0,153.0/255.0,0.0), (0.0,0.0,153.0/255.0)]
            for i in range(self.n_agent):
                agent = rendering.make_circle(radius=agent_size * scale)
                agent.set_color(*agent_clrs[i][0])
                agent_trans = rendering.Transform(
                    translation=(
                        (self.agent_positions[i][0] + 0.5) * 100 * scale,
                        (self.agent_positions[i][1] + 0.5) * 100 * scale,
                    )
                )
                agent.add_attr(agent_trans)
                self.render_agents.append(agent_trans)
                self.viewer.add_geom(agent)

            # -------------------draw agents contours
            # for i in range(self.n_agent):
            #     agent = rendering.make_circle(radius=agent_in_size * scale)
            #     agent.set_color(*agent_clrs[i][1])
            #     agent_trans = rendering.Transform(
            #         translation=(
            #             (self.agent_positions[i][0] + 0.5) * 100 * scale,
            #             (self.agent_positions[i][1] + 0.5) * 100 * scale,
            #         )
            #     )
            #     agent.add_attr(agent_trans)
            #     self.render_agents.append(agent_trans)
            #     self.viewer.add_geom(agent)

            # -------------------draw target
            tgt_l = rendering.FilledPolygon(
                [
                    (-target_w / 2.0 * scale, -target_l / 2.0 * scale),
                    (-target_w / 2.0 * scale, target_l / 2.0 * scale),
                    (target_w / 2.0 * scale, target_l / 2.0 * scale),
                    (target_w / 2.0 * scale, -target_l / 2.0 * scale),
                ]
            )
            tgt_l.set_color(*target_clrs[0])
            self.tgt_l_trans = rendering.Transform(
                translation=tuple((self.target_positions[0] + 0.5) * 100 * scale),
                rotation=np.pi / 4,
            )
            tgt_l.add_attr(self.tgt_l_trans)
            self.viewer.add_geom(tgt_l)

            tgt_r = rendering.FilledPolygon(
                [
                    (-target_w / 2.0 * scale, -target_l / 2.0 * scale),
                    (-target_w / 2.0 * scale, target_l / 2.0 * scale),
                    (target_w / 2.0 * scale, target_l / 2.0 * scale),
                    (target_w / 2.0 * scale, -target_l / 2.0 * scale),
                ]
            )
            tgt_r.set_color(*target_clrs[0])
            self.tgt_r_trans = rendering.Transform(
                translation=tuple((self.target_positions[0] + 0.5) * 100 * scale),
                rotation=-np.pi / 4,
            )
            tgt_r.add_attr(self.tgt_r_trans)
            self.viewer.add_geom(tgt_r)

            # -------------------draw target----contours
            tgt_l = rendering.FilledPolygon(
                [
                    (-target_in_w / 2.0 * scale, -target_in_l / 2.0 * scale),
                    (-target_in_w / 2.0 * scale, target_in_l / 2.0 * scale),
                    (target_in_w / 2.0 * scale, target_in_l / 2.0 * scale),
                    (target_in_w / 2.0 * scale, -target_in_l / 2.0 * scale),
                ]
            )
            tgt_l.set_color(*target_clrs[1])
            self.tgt_lc_trans = rendering.Transform(
                translation=tuple((self.target_positions[0] + 0.5) * 100 * scale),
                rotation=np.pi / 4,
            )
            tgt_l.add_attr(self.tgt_lc_trans)
            self.viewer.add_geom(tgt_l)

            tgt_r = rendering.FilledPolygon(
                [
                    (-target_in_w / 2.0 * scale, -target_in_l / 2.0 * scale),
                    (-target_in_w / 2.0 * scale, target_in_l / 2.0 * scale),
                    (target_in_w / 2.0 * scale, target_in_l / 2.0 * scale),
                    (target_in_w / 2.0 * scale, -target_in_l / 2.0 * scale),
                ]
            )
            tgt_r.set_color(*target_clrs[1])
            self.tgt_rc_trans = rendering.Transform(
                translation=tuple((self.target_positions[0] + 0.5) * 100 * scale),
                rotation=-np.pi / 4,
            )
            tgt_r.add_attr(self.tgt_rc_trans)
            self.viewer.add_geom(tgt_r)

            # -------------------draw line-----------------
            for l in range(1, self.y_len):
                line = rendering.Line((0.0, l * 100 * scale), (screen_width, l * 100 * scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.y_len):
                line = rendering.Line((l * 100 * scale, 0.0), (l * 100 * scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

        self.tgt_l_trans.set_translation(
            (self.target_positions[0][0] + 0.5) * 100 * scale,
            (self.target_positions[0][1] + 0.5) * 100 * scale,
        )
        self.tgt_r_trans.set_translation(
            (self.target_positions[0][0] + 0.5) * 100 * scale,
            (self.target_positions[0][1] + 0.5) * 100 * scale,
        )
        self.tgt_lc_trans.set_translation(
            (self.target_positions[0][0] + 0.5) * 100 * scale,
            (self.target_positions[0][1] + 0.5) * 100 * scale,
        )
        self.tgt_rc_trans.set_translation(
            (self.target_positions[0][0] + 0.5) * 100 * scale,
            (self.target_positions[0][1] + 0.5) * 100 * scale,
        )

        for i in range(self.n_agent):
            self.render_agents[i].set_translation(
                (self.agent_positions[i][0] + 0.5) * 100 * scale,
                (self.agent_positions[i][1] + 0.5) * 100 * scale,
            )
        # for i in range(self.n_agent):
        #     self.render_agents[i + 2].set_translation(
        #         (self.agent_positions[i][0] + 0.5) * 100 * scale,
        #         (self.agent_positions[i][1] + 0.5) * 100 * scale,
        #     )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

if __name__ in '__main__':
    ct = CaptureTarget(n_agent=4, n_target=1, obs_one_hot=False)
    v = ct.reset()
    print(v, np.array(v).shape)
    v = ct.get_obs()
    print(v, np.array(v).shape)
    print()
    print(ct.agent_positions)
    print(ct.target_positions)
    import math
    