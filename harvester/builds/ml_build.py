from abc import abstractmethod
from typing import List, Union, Tuple
import numpy as np

from sc2 import Result, UnitTypeId
from sharpy.managers.extensions import ChatManager
from sharpy.plans import BuildOrder
from sharpy.plans.acts import ActBase
from tactics.ml.agents import BaseMLAgent

REWARD_WIN = 1
REWARD_LOSE = 0
REWARD_TIE = 0  # Any ties are going to be strange builds anyway with 100% for example


class MlBuild(BuildOrder):
    agent: BaseMLAgent  # Initialize after init
    chatter: ChatManager

    def __init__(self, state_size: int, action_size: int, orders: List[Union[ActBase, List[ActBase]]],
                 result_multiplier: float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.reward = 0
        self.game_ended = False
        self.action: int = 0
        self.result_multiplier: float = result_multiplier
        self.last_minerals = 0
        self.action_time = -1000
        self.minimum_action_time = 1
        self.update_action_always = False  # Set this true to update bot action every step
        self.update_on_mineral_loss = True
        super().__init__(orders)

    async def start(self, knowledge: 'Knowledge'):
        await super().start(knowledge)
        self.chatter: ChatManager = self.knowledge.chat_manager

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
        self.ai.client.debug_text_screen(str(self.score), (0.00, 0.05), color, 16)
        await super().debug_draw()

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        return f'ACT{action}', (255, 255, 255)

    async def execute(self) -> bool:
        current_minerals = self.ai.minerals
        if (self.update_action_always
                or (self.update_on_mineral_loss and current_minerals < self.last_minerals)
                or self.action_time + self.minimum_action_time < self.ai.time):
            # Update action only if significant amount of time has passed or bot used minerals
            self.action_time = self.ai.time
            current_state = np.array(self.state)
            self.action = self.agent.choose_action(current_state, self.score)

        self.last_minerals = current_minerals
        await self.chat_space()
        return await super().execute()

    async def chat_space(self):
        if self.ai.time > 10:
            await self.chatter.chat_taunt_once("ml_state_space", lambda: f'State size {self.state_size}')
        if self.ai.time > 30:
            await self.chatter.chat_taunt_once("ml_action_space", lambda: f'Action size {self.action_size}')
        if self.ai.time > 40:
            await self.chatter.chat_taunt_once("ml_episodes",
                                               lambda: f'This agent has trained for {self.agent.episode} episodes')

    def on_end(self, game_result: Result):
        self.game_ended = True
        self.reward = REWARD_TIE*self.result_multiplier
        if game_result == Result.Victory:
            self.reward = REWARD_WIN*self.result_multiplier
        elif game_result == Result.Defeat:
            self.reward = REWARD_LOSE*self.result_multiplier

        self.agent.on_end(self.state, self.reward)

    def get_ml_number(self, unit_type: UnitTypeId) -> int:
        """ Calculates a funny number of building progress that's useful for machine learning"""
        units = self.cache.own(unit_type)
        normal_count = len(units)
        not_ready = units.not_ready
        not_ready_count = not_ready.amount
        normal_count = self.related_count(normal_count, unit_type)

        magic = self.unit_pending_count(unit_type) + not_ready_count
        magic += normal_count * 10

        for unit in not_ready:
            magic += unit.build_progress * 9

        return magic * 0.1  # normalize back to 1 finished building being 1
