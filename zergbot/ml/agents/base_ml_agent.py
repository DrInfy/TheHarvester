from abc import ABC, abstractmethod
from typing import List, Union


class BaseMLAgent(ABC):
    """Base machine learning agent.
    """

    def __init__(self, state_size: int, action_size: int):
        """ State size is not necessarily int, but we can only use tabular data atm """
        self.state_size = state_size
        self.action_size: int = action_size

    @abstractmethod
    def choose_action(self, state: List[Union[float, int]]):
        """Choose and return the next action.
        """
