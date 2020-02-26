import logging
from random import randint
from typing import Callable, List, Tuple

from numpy.core.multiarray import ndarray

logger = logging.getLogger(__name__)

from harvester.ml.agents import A3CAgent


class SemiRandomA3CAgent(A3CAgent):

    def __init__(self, env_name: str, state_size, action_size, log_print: Callable[[str], None] = print):
        super().__init__(env_name, state_size, action_size, log_print=log_print)

        self.start_drones = randint(12, 18)
        self.second_drones = randint(20, 30)
        self.go_hatch = randint(200, 500)

        count = randint(3, 10)
        self.action_states: List[Tuple[int, int]] = []
        time = 30
        for index in range(0, count):
            time = randint(time, time + 600)
            self.action_states.append((time, randint(0, self.action_size - 1)))

    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        self.prev_action = self.scripted_action(state)
        self.prev_state = state

        self.ep_steps += 1

        return self.prev_action

    def scripted_action(self, state: ndarray) -> int:
        action = 0
        for action_state in self.action_states:
            action = action_state[1]
            if state[0] < action_state[0]:
                break

        return action

    def scripted_action_roach(self, state: ndarray) -> int:
        if state[5] < self.start_drones or (state[9] > 1 and state[5] < self.second_drones and
                state[6] > 8):
            return 1  # Drones
        elif state[10] < 1:
            return 3  # Lings / pool first
        elif state[4] < 1 and state[7] < state[9]:
            return 2  # queens
        elif state[4] < 1 and state[3] > self.go_hatch:
            return 0  # hatcheries
        elif state[6] < 10:
            return 3  # Lings
        return 4  # Go roach
