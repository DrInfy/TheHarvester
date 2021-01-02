from random import randint
from typing import List, Union, Tuple

from numpy.core.multiarray import ndarray

from tactics.ml.agents import BaseMLAgent


class SemiScriptedAgent(BaseMLAgent):
    """
    Scripted only agent that doesn't save to model. For testing purposes
    """

    action: int  # manager needs to set this

    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.prev_action = self.scripted_action(state)
        return self.prev_action

    def scripted_action(self, state: ndarray) -> int:
        return self.action

    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)

    def on_end(self, state: List[Union[float, int]], reward: float):
        pass
