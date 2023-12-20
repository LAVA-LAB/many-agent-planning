from __future__ import annotations

import numpy as np

import numba as nb

class CoordinationGraph(object):

    def __init__(self, agent_ids : list[int], graph : list[list[int]]) -> None:
        """
        A Coordination Graph is an object to represent the coordination schemes of a set of agents. 
        Edges indicate that agents are cooperating in some way (i.e. their (action-)values overlap).
        """
        self.agent_ids : list[int] = agent_ids
        self.num_agents = len(self.agent_ids)
        self.edges : list[tuple[int]] = []
        self.vertices : set[int] = set(agent_ids)
        self.num_vertices = len(self.vertices)
        assert self.num_agents == self.num_vertices, "#Agents != #Nodes"

        self.agent_neighbors: dict[int, list[int]] = {agent_id: [] for agent_id in agent_ids}
        self.agent_edge_ids: dict[int, list[int]] = {agent_id: [] for agent_id in agent_ids}
        self.agent_edges: dict[int, list[tuple[int]]] = {agent_id: [] for agent_id in agent_ids}

        for edge_id, edge in enumerate(graph):
            assert len(edge) == 2, f"Edge {edge} is a hyperedge which is not supported currently!"
            self.edges.append(tuple(sorted(edge)))

            a1, a2 = edge
            assert a1 in self.vertices, 'Invalid coordination graph with nonexistent agent [{}]'.format(a1)
            assert a2 in self.vertices, 'Invalid coordination graph with nonexistent agent [{}]'.format(a2)

            self.agent_edge_ids[a1].append(edge_id)
            self.agent_edge_ids[a2].append(edge_id)

            self.agent_edges[a1].append((a1, a2))
            self.agent_edges[a2].append((a1, a2))

            self.agent_neighbors[a1].append(a2)
            self.agent_neighbors[a2].append(a1)
        
        self.num_edges = len(self.edges)

    def get_neighbours(self, agent_id : int) -> list[int]:
        """
        Return list of neighbours (agent ids) for some agent.
        """
        return self.agent_neighbors[agent_id]
    
    def get_edge_ids(self, agent_id : int) -> list[int]:
        """
        Return list of edges (edge id's as integers) for an agent. The edges (tuples) can be retreived from these.
        """
        return self.agent_edge_ids[agent_id]
    
    def get_all_edge_ids(self) -> list[int]:
        """
        Return list of edge ids, that is a range on the number of edges, for the entire graph.
        """
        return list(range(len(self.edges)))

    def get_edges(self, agent_id : int) -> list[tuple[int]]:
        return self.agent_edges[agent_id]
    
    def get_reverse_edges_from_edges(self, edges : list[tuple[int]]) -> list[tuple[int]]:
        return [(j, i) for (i, j) in edges]
    
    def get_edges_for_agent_from_ids(self, agent_id : int) -> list[tuple[int]]:
        """
        Return the edges (tuples) for an agent.
        """
        # Convert to np.array and back in order to do index slicing. Not very efficient of course.
        return list(map(tuple, np.array(self.edges)[self.get_edge_ids(agent_id)]))

