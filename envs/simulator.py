from __future__ import annotations

import abc

""" Abstract Simulator class. """
class Simulator(abc.ABC):

    def __init__(self, simulator) -> None:
        self._simulator = simulator
    
    def throw_error(self):
        raise NotImplementedError("This abstract method should be implemented by the child class.")

    @abc.abstractmethod
    def generative_transition(self, s, a, true_step=False):
        """ 
        s', o, r ~ G(s, a)
        """
        self.throw_error()
    
    def reset(self):
        return self._simulator.reset()
    
    def get_state(self):
        return self._simulator.state
    
    def set_state(self, state) -> None:
        self._simulator.state = state
    
    def transform(self, x):
        # Identity function by default.
        return x

    @abc.abstractmethod
    def get_possible_actions(self, *args):
        self.throw_error()
    
    @abc.abstractmethod
    def get_random_action(self, state=None):
        self.throw_error()
    
    @abc.abstractmethod
    def sample_initial(self):
        self.throw_error()

    def step(self, s, a, **kwargs):
        """
        Models gym environment steps:
        s, a -> next_state, observation, reward, info
        """
        s_, o, r, info = self.generative_transition(s, a, **kwargs)
        if isinstance(info, dict):
            step = info['n_steps']
        elif isinstance(info, int):
            step = info
        else:
            raise ValueError(f"Unexpected information value returned from generative transition: {info}")
        return s_, o, r, step