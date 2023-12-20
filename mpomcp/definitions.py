from __future__ import annotations

from dataclasses import dataclass
import random

class TreeNode:
    """
    Base class for search tree nodes.
    """
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        return self.children.get(key, None)
    def __setitem__(self, key, value):
        self.children[key] = value
    def __contains__(self, key):
        return key in self.children

class QNode(TreeNode):
    """
    The action nodes of the tree.
    """
    def __init__(self, num_visits, value, prob=1, parent=None):
        " Action node "
        self.num_visits = num_visits
        self.value = value
        self.parent = parent
        self.prob = prob
        self.children = {}  # o -> VNode
    def __str__(self):
        return ("QNode") + "(%.3f, %.3f | %s)" % (self.num_visits,
                                                  self.value,
                                                  str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class VNode(TreeNode):
    """
    The observation nodes of the tree.
    """
    def __init__(self, num_visits, hidden_state = None, parent : QNode = None, belief = None, factored_statistics = None, rho = None, **kwargs):
        " Observation / history node "
        self.num_visits = num_visits
        self.parent = parent
        self.children = {} # a -> QNode
        self.hidden_state = hidden_state
        self.belief = belief
        self.rho = rho
        self.factored_statistics = factored_statistics

    def __str__(self):
        return ("VNode") + "(%.3f, %.3f | %s)" % (self.num_visits,
                                                  self.value,
                                                  str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self[action].value))

    def argmax(self):
        """argmax(VNode self)
        Returns the action of the child with highest value"""
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].value > best_value:
                best_action = action
                best_value = self[action].value
        return best_action

    @property
    def value(self):
        best_action = max(self.children, key=lambda action: self.children[action].value)
        return self.children[best_action].value


class RootVNode(VNode):
    """
    The root node of the search tree.
    """
    def __init__(self, num_visits, history):
        VNode.__init__(self, num_visits)
        self.history = history

    @classmethod
    def from_vnode(cls, vnode, history):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, history)
        rootnode.children = vnode.children
        rootnode.hidden_state = vnode.hidden_state
        rootnode.parent = vnode.parent
        rootnode.belief = vnode.belief
        rootnode.rho = vnode.rho
        rootnode.factored_statistics = vnode.factored_statistics
        return rootnode

@dataclass
class FactoredStatistic:
    """
    Helper class to represents factored statistics inside the tree nodes.
    """
    num_visits : int
    value : float

    def __init__(self, num_visits = 0, value = 0) -> None:
        self.num_visits = num_visits
        self.value = value

@dataclass
class Factor:
    """
    Class that represents a factor in a coordination graph for building search trees.
    """
    factor_id : int
    agent_ids : list[int]
    tree : TreeNode
    history : list[tuple]
    belief : list

    def __init__(self, id : int, tree : TreeNode = None, agent_ids : list[int] = None, history=[], belief=None):
        self.factor_id = id
        self.agent_ids = agent_ids
        self.tree = tree
        self.history = history
        self.belief = belief
