from __future__ import annotations

from graphs.cg import CoordinationGraph

import random

class FactoredSimulators:

    @staticmethod
    def factored_to_total(graph : CoordinationGraph, act_or_obs : list[list[int]], agents_per_factor = 2) -> list[int]:
        """Inverse of total_to_factored, needs some assumption on agents per factor.

        Parameters
        ----------
        graph : CoordinationGraph
            _description_
        act_or_obs : list[list[int]]
            _description_
        agents_per_factor : int, optional
            _description_, by default 2

        Returns
        -------
        list[int]
            Flattened joint action as array with actions of individual agents.

        Raises
        ------
        NotImplementedError
            _description_
        """
        if agents_per_factor != 2:
            raise NotImplementedError()

        joint_list = [None] * graph.num_agents
        try:
            for (i, j), (ai, aj) in zip(graph.edges, act_or_obs):
                if joint_list[i] is not None:
                    assert ai == joint_list[i]
                else:
                    joint_list[i] = ai

                if joint_list[j] is not None:
                    assert aj == joint_list[j]
                else:
                    joint_list[j] = aj
        except AssertionError as ae:
            print(graph.edges)
            print(act_or_obs)
            print(joint_list)
            print((ai, i))
            print((aj, j))
            raise ae

        return joint_list

    @staticmethod
    def total_to_factored(graph : CoordinationGraph, act_or_obs : list[int]) -> list[tuple[int]]:
        """Converts a vector of actions or observations to a factored representation

        EXAMPLE:
        Three agent two factor case:
        obs = [a, b, c]
        graph = a <-> b <-> c
        factored obs <- [[a,b], [b, c]]

        Parameters
        ----------
        graph : CoordinationGraph
            The graph representing the coordination scheme and accompanying factors.
        act_or_obs : list[int]
            The vector of actions/observations.

        Returns
        -------
        list[tuple[int]]
            Vector of factored actions/observations, 
        """
        factored_obs = [None] * graph.num_edges

        for id, edge in enumerate(graph.edges):
            # Agent 1 and agent 2
            n1, n2 = edge
            factored_obs[id] = (act_or_obs[n1], act_or_obs[n2])

        assert None not in factored_obs

        return factored_obs
    
    @staticmethod
    def generate_random_dense_graph(num_agents : int):
        random_edges = [(i, j) for i in range(num_agents) for j in range(i, num_agents) if i != j]
        return CoordinationGraph(list(range(num_agents)), random_edges)
    
    @staticmethod
    def generate_random_graph(num_agents : int, edge_degree = 2) -> CoordinationGraph:
        """Generate a graph with random connection without replacement of agents of certain degree. Every agent only appears in a single edge.

        Parameters
        ----------
        num_agents : int
            Number of agents to create graph for
        edge_degree : int, optional
            A.k.a. agents per factor/edge, by default 2

        Returns
        -------
        CoordinationGraph
            Graph containing the random teams.
        """
        agents = list(range(num_agents))
        random.shuffle(agents)
        random_edges = list(agents[i:i+edge_degree] for i in range(0, num_agents, edge_degree))
        return CoordinationGraph(list(range(num_agents)), random_edges)
