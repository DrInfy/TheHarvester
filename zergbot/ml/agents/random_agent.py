import random
from typing import List, Union

from zergbot.ml.agents import BaseMLAgent
from numpy.core.multiarray import ndarray


class RandomAgent(BaseMLAgent):
    """Random Agent that takes random actions in the game.
    """

    def choose_action(self, state: ndarray, reward: float) -> int:
        return random.randint(0, self.action_size - 1)

    def on_end(self, state: List[Union[float, int]], reward: float):
        pass
