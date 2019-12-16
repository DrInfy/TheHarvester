from math import floor
import random


class BaseMLAgent:
    """Base machine learning agent.
    """
    def choose_action(self, state):
        """Choose and return the next action.
        """
        pass


class RandomAgent(BaseMLAgent):
    """Random Agent that takes random actions in the game.
    """

    def choose_action(self, state):
        return random.randint(0, 1)


class SemiScriptedAgent(BaseMLAgent):
    """Semi Scripted Agent that takes semi-random actions in the game until it has 50 workers,
     at which point it creates army.
    """

    def choose_action(self, state):
        if state[1] >= 50:
            return 1
        else:
            return floor(state[0] / 90) % 2


class A3CAgent(BaseMLAgent):
    """A3C machine learning agent.
    """

    def choose_action(self, state):
        pass  # todo
