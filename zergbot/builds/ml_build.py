from abc import abstractmethod
from typing import List, Union, Tuple

from sc2 import Result
from sharpy.plans import BuildOrder
from sharpy.plans.acts import ActBase

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_TIE = 0


class MlBuild(BuildOrder):
    def __init__(self, state_size: int, action_size: int, orders: List[Union[ActBase, List[ActBase]]]):
        self.action_size = action_size
        self.state_size = state_size
        self.reward = 0
        self.action: int = 0
        super().__init__(orders)

    @abstractmethod
    def state(self) -> List[int]:
        pass

    async def debug_draw(self):
        action_name, color = self.get_action_name_color(self.action)
        self.ai.client.debug_text_screen(action_name, (0.01, 0.01), color, 16)

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        return f'ACT{action}', (255, 255, 255)

    def on_end(self, game_result: Result):
        self.reward = REWARD_TIE
        if game_result == Result.Victory:
            self.reward = REWARD_WIN
        elif game_result == Result.Defeat:
            self.reward = REWARD_LOSE
