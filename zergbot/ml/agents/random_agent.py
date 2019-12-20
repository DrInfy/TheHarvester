import random
from typing import List, Union

from zergbot.ml.agents import BaseMLAgent


class RandomAgent(BaseMLAgent):
    """Random Agent that takes random actions in the game.
    """

    def choose_action(self, state: List[Union[float, int]]):
        return random.randint(0, self.action_size - 1)
