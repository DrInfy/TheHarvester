from abc import ABC, abstractmethod

from tactics.ml.agents import BaseMLAgent


class BaseEnv(ABC):
    def __init__(self, agent: BaseMLAgent) -> None:
        super().__init__()
        self.agent = agent

    @abstractmethod
    def run(self):
        pass
