from math import floor
from typing import List, Union

from zergbot.ml.agents import BaseMLAgent


class SemiScriptedAgent(BaseMLAgent):
    """Semi Scripted Agent that takes semi-random actions in the game until it has 50 workers,
     at which point it creates army.
    """

    def choose_action(self, state: List[Union[float, int]]):
        if state[1] >= 50:
            return 1
        else:
            return floor(state[0] / 90) % 2
