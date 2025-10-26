from __future__ import annotations

import sys
sys.path.append('/home/maris/ellis/pomcp/mpomcp/')

import math
import random
import copy

from mpomcp.definitions import QNode, VNode
from graphs.cg import CoordinationGraph

from tqdm import tqdm
import numpy as np

import numba as nb

num_iterations = 100

def get_utility_value(actions : list[int], edge_qs : dict[int, np.ndarray], graph : CoordinationGraph) -> float:
    q_sum = 0
    for edge_id in graph.get_all_edge_ids():
        n1, n2 = graph.edges[edge_id]
        edge_q = edge_qs[edge_id][actions[n1], actions[n2]].max()
        q_sum += edge_q
    return q_sum

def max_plus_final(graph : CoordinationGraph, max_iterations : int, roots : list[VNode], action_size_f, exploration_const, agent_ordering : list[int] = None, use_ucb = True, random_tie_break = True, using_NN = False):
    actions_per_agent = action_size_f(0) # TODO: this assumes homogeneity..

    messages_fwd = np.zeros((graph.num_edges, actions_per_agent))
    messages_bwd = np.zeros_like(messages_fwd)

    final_q_values = np.zeros((graph.num_agents, actions_per_agent))

    edge_qs = np.array([list(root.children.values()) for root in roots]).reshape((graph.num_edges, actions_per_agent, actions_per_agent))
    edge_n = np.array(list(map(lambda x : x.num_visits, roots)), dtype=int)

    # TODO: in FactoredValueMCTS they use individual agent utils (estimated). They claim by emperical results that it improves performance.

    def message_passing(messages_fwd, messages_bwd, action_size_f, edge_qs, c, do_edge_explore=False, do_node_explore=False, message_norm = True):
        # Make copy of previous messages. Everything that does a real deep copy is more effective than the more general deepcopy.
        old_messages_fwd = np.array(messages_fwd)
        old_messages_bwd = np.array(messages_bwd)

        # IMPORTANT! WE ASSUME THAT ACTION NUMBERS HAVE A ONE TO ONE CORRESPONDENCE TO ACTION INDICES

        graph_edges = [e for e in graph.edges]
        idxs = list(range(graph.num_edges))
        graph_edges.extend(graph.get_reverse_edges_from_edges(graph.edges))
        idxs.extend(idxs)

        # for edge_idx, (i, j) in enumerate(graph_edges):
        for edge_idx, (i, j) in zip(idxs, graph_edges):
            # FORWARD PASS
            for aj in range(action_size_f(j)): # TODO: make this non-static
                fwd_message_values = np.zeros(action_size_f(i))
                for ai in range(action_size_f(i)): # TODO: make this non-static
                    q_value = 0 # Individual agent utility, not used for now.
                    edge_q = edge_qs[edge_idx][ai][aj]
                    # UCB edge exploration. Also divide Q-value by total number of edges.
                    ucb_or_edge_q = edge_q.value
                    ucb_or_edge_q /= graph.num_edges
                    if do_edge_explore:
                        ucb_or_edge_q += c * math.sqrt((math.log(edge_n[edge_idx] + 1)) / (edge_q.num_visits + 1))
                    fwd_message_values[ai] = q_value - old_messages_bwd[edge_idx, ai] + ucb_or_edge_q
                messages_fwd[edge_idx, aj] = max(fwd_message_values)

            # BACKWARDS PASS (make re-usable function for this maybe?)
            for ai in range(action_size_f(i)): # TODO: make this non-static
                bwd_message_values = np.zeros(action_size_f(j))
                for aj in range(action_size_f(j)): # TODO: make this non-static
                    q_value = 0 # Individual agent utility, not used for now.
                    edge_q = edge_qs[edge_idx][ai][aj]
                    # UCB edge exploration. Also divide Q-value by total number of edges.
                    ucb_or_edge_q = edge_q.value 
                    ucb_or_edge_q /= graph.num_edges
                    if do_edge_explore:
                        ucb_or_edge_q += c * math.sqrt((math.log(edge_n[edge_idx] + 1)) / (edge_q.num_visits + 1))
                    bwd_message_values[aj] = q_value - old_messages_fwd[edge_idx, aj] + ucb_or_edge_q
                messages_bwd[edge_idx, ai] = max(bwd_message_values)
            
            if message_norm:
                messages_fwd[edge_idx, :] -= sum(messages_fwd[edge_idx, :]) / len(messages_fwd[edge_idx, :])
                messages_bwd[edge_idx, :] -= sum(messages_bwd[edge_idx, :]) / len(messages_bwd[edge_idx, :])
        
        fwd_norm = np.linalg.norm(messages_fwd - old_messages_fwd)
        bwd_norm = np.linalg.norm(messages_bwd - old_messages_bwd)

        return messages_fwd, messages_bwd, fwd_norm, bwd_norm

    for _ in range(max_iterations):
        # BEGIN MESSAGEPASSING
        messages_fwd, messages_bwd, fwd_norm, bwd_norm = message_passing(messages_fwd, messages_bwd, action_size_f, edge_qs, exploration_const, do_edge_explore=False, do_node_explore=False)
        # END MESSAGEPASSING
        
        if math.isclose(fwd_norm, 0.0) and math.isclose(bwd_norm, 0.0):
            # We have converged
            break

    # if edge exploration -> do single round of messagepassing WITH exploration bonus (UCB).
    if use_ucb:
        messages_fwd, messages_bwd, fwd_norm, bwd_norm = message_passing(messages_fwd, messages_bwd, action_size_f, edge_qs, exploration_const, do_edge_explore=True, do_node_explore=False)

    best_action = np.zeros(graph.num_agents)

    # if not using individual agent utils, then:
    final_q_values = np.zeros((graph.num_agents, actions_per_agent))

    for agent_id in graph.agent_ids:
        neighbours = set(graph.get_neighbours(agent_id))

        agent_edges = [e for e in graph.get_edges(agent_id)]
        idxs = list(range(len(agent_edges)))
        agent_edges.extend(graph.get_reverse_edges_from_edges(agent_edges))
        idxs.extend(idxs)

        # for edge_idx, (i, j) in enumerate(graph.get_edges(agent_id))
        for edge_idx, (i, j) in zip(idxs, agent_edges):
            if i == agent_id:
                assert j in neighbours, f"Edge ({(i, j)}) should exist! Agent ID:, {agent_id}, Edges: {edges}, Neighbours: {neighbours}"
                # We -> Them
                final_q_values[agent_id, :] += messages_bwd[edge_idx, :]
                # They -> Us
                final_q_values[j, :] += messages_fwd[edge_idx, :]
            elif j == agent_id:
                assert i in neighbours, f"Edge ({(i, j)}) should exist! Agent ID:, {agent_id}, Edges: {edges}, Neighbours: {neighbours}"
                # We -> Them
                final_q_values[agent_id, :] += messages_bwd[edge_idx, :]
                # They -> Us
                final_q_values[i, :] += messages_fwd[edge_idx, :]
            else:
                raise ValueError(f"Edge ({(i, j)}) should exist! Agent ID:, {agent_id}, Edges: {edges}, Neighbours: {neighbours}")

    for agent_id in graph.agent_ids:
        # if node exploration do the same but with exploration bonus (UCB)
        expected_q_values = final_q_values[agent_id][:]

        if random_tie_break:
            best_action_idx = random.choice(np.flatnonzero(expected_q_values == expected_q_values.max()))
        else:
            best_action_idx = expected_q_values.argmax()

        # In our case the best action index is also the best action
        best_action[agent_id] = best_action_idx
    
    return best_action

@nb.jit(nopython=True)
def message_passing_jit(messages : np.ndarray, old_msg : np.ndarray, agent_edges : dict[int, list[tuple[int]]], agent_neighbours : dict[int, list[int]], edge_qs : dict[int, np.ndarray], agent_ordering : list[int]):
    for agent_id in agent_ordering:
        edges = agent_edges[agent_id]
        for edge_idx, (i, j) in enumerate(edges):
            if agent_id == i:
                neighbour = j
                edge_q = edge_qs[edge_idx]
            elif agent_id == j:
                neighbour = i
                edge_q = edge_qs[edge_idx].T
            else:
                raise ValueError()

            neighbours = [x for x in agent_neighbours[agent_id] if x != neighbour]
            # Don't sum over the neighbour that we're computing the current message for.
            if len(neighbours) > 0:
                neighbour_sum = np.atleast_2d(messages[neighbours, agent_id].sum(axis=0))
                message = (edge_q + neighbour_sum).max(axis=0)
            else:
                # Don't add anything if there are no neighbours at this stage.
                message = edge_q.max(axis=0)
            messages[agent_id][neighbour] = - old_msg[agent_id][neighbour] + message
    return messages, np.linalg.norm(messages - old_msg)

def message_passing(messages : np.ndarray, graph, edge_qs : dict[int, np.ndarray], agent_ordering):
    old_msg = messages.copy()
    for agent_id in agent_ordering:

        edges = graph.get_edges(agent_id)

        for edge_idx, (i, j) in enumerate(edges):
            if agent_id == i:
                neighbour = j
                edge_q = edge_qs[edge_idx]
            elif agent_id == j:
                neighbour = i
                edge_q = edge_qs[edge_idx].T
            else:
                raise ValueError()

            neighbours = [x for x in graph.get_neighbours(agent_id) if x != neighbour]
            # Don't sum over the neighbour that we're computing the current message for.
            if len(neighbours) > 0:
                neighbour_sum = np.atleast_2d(messages[neighbours, agent_id].sum(axis=0))
                message = (edge_q + neighbour_sum).max(axis=0)
            else:
                # Don't add anything if there are no neighbours at this stage.
                message = edge_q.max(axis=0)
            messages[agent_id][neighbour] = - old_msg[agent_id][neighbour] + message
        # Message normalisation.
        # messages[agent_id, ...] -= sum(messages[agent_id, ...]) / len(messages[agent_id, ...])
    return messages, np.linalg.norm(messages - old_msg)

def compute_optimal_local_actions(graph, messages, random_tie_break):
    local_actions = [None] * graph.num_agents
    for agent_id in graph.agent_ids:
        ns = graph.get_neighbours(agent_id)
        g = messages[ns, agent_id].sum(axis=0)
        max_g = g.max()
        local_actions[agent_id] = np.random.choice(np.where(g == max_g)[0]) if random_tie_break else np.argmax(g)
    return local_actions

def max_plus_ucb_final(graph : CoordinationGraph, max_iterations : int, edge_qs : dict[int, np.ndarray], edge_qs_ucb : dict[int, np.ndarray], action_size_f, agent_ordering:list[int]=None, use_ucb=True, debug=False, random_tie_break=True, jit=False) -> list[int]:
    """
    Centralised/Iterative Max Plus algorithm for action selection using message passing. This version requires both an edge_q dictionary with and without exploration bonus as arguments. 

    Parameters
    ----------
    graph : CoordinationGraph
        The graph structure the agents resides on.
    max_iterations : int
        maximum number of iterations the algorithm will pass messages before selecting an action.
    edge_qs : dict[int, np.ndarray]
        dictionary with the conditional (vanilla) Q-values per edge
    edge_qs_ucb : dict[int, np.ndarray]
        dictionary with the conditional ((P)UCB/exploration bonus included) Q-values per edge
    verbose : bool, optional
        _description_, by default False
    iterations : _type_, optional
        _description_, by default None
    use_ucb : bool, optional
        _description_, by default True
    debug : bool, optional
        _description_, by default False
    random_tie_break : bool, optional
        _description_, by default True

    Returns
    -------
    list[int]
        list of length with individual actions equal to the number of agents.

    Raises
    ------
    ValueError
        Only raised if an edge is found that does not contain the agent id that the edges were queried for. Most probably a problem in the CoordinationGraph class.
    """    
    
    actions_per_agent = action_size_f(0)

    # ai -> aj
    messages = np.zeros((graph.num_agents, graph.num_agents, actions_per_agent))
    agent_order = graph.agent_ids if agent_ordering is None else agent_ordering

    actions = [None] * graph.num_agents
    best_actions = actions
    total_utility = -np.inf
        
    for iters in range(max_iterations):
        # FORWARD pass
        messages, norm = message_passing(messages, graph, edge_qs_ucb if use_ucb else edge_qs, agent_ordering=reversed(agent_order) if (iters % 2 == 1) else agent_order) # do reverse pass on odd (second) iterations.
    
        # Anytime extension.
        actions = compute_optimal_local_actions(graph, messages, random_tie_break)
        val = get_utility_value(actions, edge_qs_ucb if use_ucb else edge_qs, graph)
        if val > total_utility:
            total_utility = val
            best_actions = actions

        if iters > 1 and math.isclose(norm, 0):
            if debug: print(f"Max-Plus converged in {iters} iterations.")
            if debug: print("MSGN:", messages.sum())
            break
    else:
        if debug: print(f"Max-Plus reached maxmimum ({iters}) iterations. Norm = {norm}.")
        if debug: print("MSGN:", messages.sum())

    
    assert None not in set(best_actions)
    
    return best_actions
