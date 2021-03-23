from abc import ABC, abstractmethod
from typing import List, Union

from numpy.core.multiarray import ndarray


class BaseMLAgent(ABC):
    """Base machine learning agent.
    """

    def __init__(self, state_size: int, action_size: int):
        """ State size is not necessarily int, but we can only use tabular data atm """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.episode: int = 0


    @abstractmethod
    def on_start(self, state: List[Union[float, int]]):
        """Perform any starting tasks
        """

    @abstractmethod
    def choose_action(self, state: ndarray, reward: float) -> int:
        """
        Choose and return the next action.
        :param state: numpy array
        :param reward: float as the reward value
        :return: action type integer
        """

    @abstractmethod
    def on_end(self, state: List[Union[float, int]], reward: float):
        """Perform any ending tasks
        """
