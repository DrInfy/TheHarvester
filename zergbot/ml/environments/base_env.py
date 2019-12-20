from abc import ABC, abstractmethod
from typing import Callable


class BaseEnv(ABC):
    def __init__(self, on_step: Callable, on_end: Callable) -> None:
        super().__init__()
        self.on_step = on_step
        self.on_end = on_end

    @abstractmethod
    def start(self):
        pass
