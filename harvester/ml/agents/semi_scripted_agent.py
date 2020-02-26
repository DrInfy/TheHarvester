from math import floor
from random import randint
from typing import List, Union, Tuple

from numpy.core.multiarray import ndarray

from harvester.ml.agents import BaseMLAgent


class SemiScriptedAgent(BaseMLAgent):
    """
    Semi Scripted Agent that uses one build and then switches it to another.
    Attempt to help exploring.
    self.ai.time must be state 0
    """

    def __init__(self, state_size: int, action_size: int):
        self.action_size = action_size
        count = randint(3, 10)
        self.action_states: List[Tuple[int, int]] = []
        time = 30
        for index in range(0, count):
            time = randint(time, time + 600)
            self.action_states.append((time, randint(0, self.action_size - 1)))

        super().__init__(state_size, action_size)

    def choose_action(self, state: ndarray, reward: float) -> int:
        action = 0
        for action_state in self.action_states:
            action = action_state[1]
            if state[0] < action_state[0]:
                break

        return action

    def on_end(self, state: List[Union[float, int]], reward: float):
        pass
