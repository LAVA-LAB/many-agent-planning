from __future__ import annotations

"""
Inspired by the implementation from BOSS (https://github.com/rail-cwru/boss).
"""

from graphs.cg import CoordinationGraph

import numpy as np

import time

class EliminationTree(object):

    def __init__(self, graph: CoordinationGraph):
        """
        Initialise the elimination tree from the coordination graph.
        """
        # Mutable tree as list for elimination add adding of leaf nodes.
        tree = [(eid, list(edge)) for eid, edge in (zip(graph.get_all_edge_ids(), graph.edges))]
        # Get all agents that are connected with edges. We should not have non-connected agents.
        agents = [aid for aid in graph.agent_ids if len(graph.agent_edges[aid]) > 0]
        # Given an agent, have the order of the leaves subordinate to it
        self.agent_leaforder = {}
        # Given an agent, have the order of the data necessary to organize the subordinate leaf data
        self.agent_subleaves = {}
        # First agent to eliminate should be of least width
        self.agentorder = self.width_sort(agents, tree)
        # Construct the elimination tree
        for i, agent_id in enumerate(self.agentorder):
            # Determine the subordinate leaves of the agent
            neighbours = [(label, list(edge)) for label, edge in tree if agent_id in edge]
            leaves = []
            for node in neighbours:
                # Eliminate neighbour
                tree.remove(node)
                edge = node[1]
                # Add all agents that were pruned (without adding self).
                leaves.extend([subleaf for subleaf in edge if subleaf not in leaves and subleaf != agent_id])

            # Agent + leaves, for combining q-values
            plus_leaves = [agent_id] + leaves
            # Construct the leaf data manipulation objects
            neighbor_data = []  # (neighbor node id, slice_ext, transpose_order), or transform-func?
            for label, subleaves in neighbours:
                # Needs to know which node each neighbor is
                slice_ext = tuple([...] + [None] * (len(plus_leaves) - len(subleaves)))
                extended_subleaves = subleaves + [leaf for leaf in plus_leaves if leaf not in subleaves]
                transpose_order = [extended_subleaves.index(leaf) for leaf in plus_leaves]
                neighbor_data.append((label, slice_ext, transpose_order))
            # Append data to the tree and node
            tree.append((-agent_id-1, leaves))
            self.agent_leaforder[agent_id] = leaves
            self.agent_subleaves[agent_id] = neighbor_data
            if i < len(agents) - 1:
                self.agentorder = self.agentorder[:i + 1] + self.width_sort(self.agentorder[i + 1:], tree)
        # print(neighbours, self.agent_subleaves, self.agent_leaforder)
        # print(len(neighbours[0]), len(self.agent_subleaves[0]), len(self.agent_leaforder[0]))

    def size(self) -> int:
        """
        Size of the computed elimination tree. If nodes are removed then they are replaced with subleaves conditioned on the actions of the removed node.
        """
        return sum([len(self.agent_subleaves[agent_id]) for agent_id in self.agentorder])

    def width_sort(self, agents : list[int], tree : list[tuple]) -> list[int]:
        """
        Sort the agents of the graphs according to their degree.
        """
        # TODO actually sort according to width instead of assuming homogeneity
        degrees = []
        for aid in agents:
            degrees.append(len([(label, list(edge)) for label, edge in tree if aid in edge]))
        _, deg_sorted = zip(*sorted(zip(degrees, agents)))
        return deg_sorted

    def agent_elimination(self, eqs : dict[int, list], random_tiebreak=True, timeout = 600, **kwargs) -> list[int]:
        """
        Run variable elimination on the elimination tree and return the optimal result.
        """
        tik = time.time()
        agent_action_funcs = {}
        # For each agent
        for aid in self.agentorder:
            if time.time() - tik > timeout:
                return [None] * len(self.agentorder)
            neighbors = self.agent_subleaves[aid]
            # Calculate the expanded q-values (with max-q information from subordinate nodes)
            label, ext, order = neighbors[0]
            all_q = eqs[label][ext].transpose(order)
            for label, ext, order in neighbors[1:]:
                if time.time() - tik > timeout:
                    return [None] * len(self.agentorder)
                try:
                    all_q = all_q + eqs[label][ext].transpose(order)
                except (np.AxisError, KeyError) as e:
                    print(aid, label, ext, order)
                    print(list(eqs.keys()))
                    raise e
            # Take the rowmax for the max q values for this node to pass up the tree and take the argmax for the agent action function
            rowmax = all_q.max(axis=0)
            eqs[-aid - 1] = rowmax
            agent_action_funcs[aid] = np.argmax(np.random.random(all_q.shape) * (all_q==rowmax.max()), axis=0) if random_tiebreak else all_q.argmax(axis=0)
        # Determine action map
        action_map = [None] * len(self.agentorder)
        for aid in reversed(self.agentorder):
            if time.time() - tik > timeout:
                return action_map
            # Given the chosen action from upper agents, choose the actions for the lower agents
            indexing = tuple([action_map[leaf] for leaf in self.agent_leaforder[aid]])
            action_map[aid] = int(agent_action_funcs[aid][indexing])
        return action_map
