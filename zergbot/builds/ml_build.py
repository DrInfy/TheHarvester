from abc import abstractmethod
from typing import List, Union, Tuple

from sc2 import Result
from sharpy.plans import BuildOrder
from sharpy.plans.acts import ActBase
from zergbot.ml.agents import BaseMLAgent

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_TIE = 0


class MlBuild(BuildOrder):
    def __init__(self, agent: BaseMLAgent, state_size: int, action_size: int,  orders: List[Union[ActBase, List[ActBase]]]):
        self.action_size = action_size
        self.state_size = state_size
        self.agent = agent
        self.reward = 0
        self.game_ended = False
        self.action: int = 0
        super().__init__(orders)

    @property
    @abstractmethod
    def state(self) -> List[Union[int, float]]:
        pass

    @property
    def score(self) -> float:
        return self.reward

    async def debug_draw(self):
        action_name, color = self.get_action_name_color(self.action)
        self.ai.client.debug_text_screen(action_name, (0.01, 0.01), color, 16)

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        return f'ACT{action}', (255, 255, 255)

    async def execute(self) -> bool:
        current_state = self.state
        self.action = self.agent.choose_action(current_state, self.score)
        return await super().execute()

    def on_end(self, game_result: Result):
        self.game_ended = True
        self.reward = REWARD_TIE
        if game_result == Result.Victory:
            self.reward = REWARD_WIN
        elif game_result == Result.Defeat:
            self.reward = REWARD_LOSE

        self.agent.on_end(self.state, self.reward)
