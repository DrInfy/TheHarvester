import logging
from random import randint
from typing import Callable, List, Tuple

from numpy.core.multiarray import ndarray

logger = logging.getLogger(__name__)

from tactics.ml.agents import A3CAgent


class SemiRandomA3CAgent(A3CAgent):
    action: int  # manager needs to set this

    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        self.prev_action = self.scripted_action(state)
        self.prev_state = state

        self.ep_steps += 1

        return self.prev_action

    def scripted_action(self, state: ndarray) -> int:
        return self.action
