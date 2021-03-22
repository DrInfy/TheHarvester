from abc import abstractmethod
from typing import List, Union, Tuple, Callable
import numpy as np

from sc2 import Result, UnitTypeId
from sc2.constants import EQUIVALENTS_FOR_TECH_PROGRESS
from sc2.ids.upgrade_id import UpgradeId
from sharpy.managers.extensions import ChatManager
from sharpy.plans import BuildOrder
from sharpy.plans.acts import ActBase
from tactics.ml.agents import BaseMLAgent

REWARD_WIN = 1
REWARD_LOSE = 0
REWARD_TIE = 0  # Any ties are going to be strange builds anyway with 100% for example


class StateFunc:  # Statefunction
    def __init__(self, name: str, func: Callable[[], float]) -> None:
        self.name = name
        self.f = func


class MlBuild(BuildOrder):
    agent: BaseMLAgent  # Initialize after init
    chatter: ChatManager

    def __init__(
        self,
        state_size: int,
        action_size: int,
        orders: Union[
            Union[ActBase, list, Callable[["Knowledge"], bool]],
            List[Union[ActBase, list, Callable[["Knowledge"], bool]]],
        ],
        result_multiplier: float = 1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.reward = 0
        self.last_reward = 0
        self.game_ended = False
        self.action: int = 0
        self.result_multiplier: float = result_multiplier
        self.last_minerals = 0
        self.action_time = -1000
        self.minimum_action_time = 1
        self.update_action_always = False  # Set this true to update bot action every step
        self.update_on_mineral_loss = True
        self.use_difference_reward = True
        super().__init__(orders)

    async def start(self, knowledge: "Knowledge"):
        await super().start(knowledge)
        self.chatter: ChatManager = self.knowledge.get_required_manager(ChatManager)

    @property
    @abstractmethod
    def state(self) -> List[Union[int, float]]:
        pass

    def calc_reward(self):
        self.reward = 0

    @property
    def score(self) -> float:
        if self.use_difference_reward:
            return self.reward - self.last_reward
        return self.reward

    async def debug_draw(self):
        action_name, color = self.get_action_name_color(self.action)
        self.ai.client.debug_text_screen(action_name, (0.01, 0.01), color, 16)
        self.ai.client.debug_text_screen(str(self.score), (0.00, 0.05), color, 16)
        await super().debug_draw()

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        return f"ACT{action}", (255, 255, 255)

    @property
    def ready_to_update(self):
        current_minerals = self.ai.minerals
        if (
            self.update_action_always
            or (self.update_on_mineral_loss and current_minerals < self.last_minerals)
            or self.action_time + self.minimum_action_time < self.ai.time
        ):
            # Update action only if significant amount of time has passed or bot used minerals
            self.action_time = self.ai.time
            self.last_minerals = current_minerals
            return True

        return False

    async def execute(self) -> bool:
        await self.chat_space()
        return await super().execute()

    async def chat_space(self):
        if self.ai.time > 10:
            await self.chatter.chat_taunt_once("ml_state_space", lambda: f"State size {self.state_size}")
        if self.ai.time > 20:
            await self.chatter.chat_taunt_once("ml_action_space", lambda: f"Action size {self.action_size}")
        if self.ai.time > 25:
            await self.chatter.chat_taunt_once("ml_actions", self.write_actions)
        if self.ai.time > 40:
            await self.chatter.chat_taunt_once(
                "ml_episodes", lambda: f"This agent has trained for {self.agent.episode} episodes"
            )

    def write_actions(self) -> str:
        text = "Possible actions: "
        for i in range(0, self.action_size):
            text += str(i) + ": " + self.get_action_name_color(i)[0] + " "
        return text

    def on_end(self, game_result: Result):
        self.game_ended = True
        self.reward = REWARD_TIE * self.result_multiplier
        if game_result == Result.Victory:
            self.reward = REWARD_WIN * self.result_multiplier
        elif game_result == Result.Defeat:
            self.reward = REWARD_LOSE * self.result_multiplier

        self.agent.on_end(self.state, self.reward)

    def get_ml_upgrade_progress(self, upgrades: List[UpgradeId]):
        value = 0
        for upgrade in upgrades:
            if upgrade in self.knowledge.version_manager.disabled_upgrades:
                return 0
            tmp = self.ai.already_pending_upgrade(upgrade)
            if 0 < tmp < 1:
                tmp = 0.25 + tmp * 0.5

            value += tmp
        return value

    def get_ml_number(self, unit_type: UnitTypeId) -> float:
        """ Calculates a funny number of building progress that's useful for machine learning"""
        units = self.cache.own(unit_type)
        normal_count = len(units.ready)
        not_ready = units.not_ready
        not_ready_count = not_ready.amount
        if unit_type in EQUIVALENTS_FOR_TECH_PROGRESS:
            normal_count += self.cache.own(EQUIVALENTS_FOR_TECH_PROGRESS[unit_type]).ready.amount

        magic = self.unit_pending_count(unit_type) + not_ready_count
        magic += normal_count * 10

        for unit in not_ready:
            magic += unit.build_progress * 9

        return magic * 0.1  # normalize back to 1 finished building being 1
