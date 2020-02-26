import logging
from random import randint

from numpy.core.multiarray import ndarray

logger = logging.getLogger(__name__)

from harvester.ml.agents import A3CAgent


class RandomA3CAgent(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        self.prev_action = randint(0, self.action_size - 1)
        self.prev_state = state

        self.ep_steps += 1

        return self.prev_action
