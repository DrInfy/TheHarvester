from abc import abstractmethod, ABC
from typing import Dict, Callable, Optional
import numpy as np
from sc2 import Result
from sc2.position import Point2
from sharpy.managers import ManagerBase
from tactics.ml.agents import *
from tactics.ml.ml_build import MlBuild


class BaseAgentManager(ManagerBase):
    agent: BaseMLAgent
    build: MlBuild

    def __init__(self, agent: str, build_str: str, build: MlBuild) -> None:
        super().__init__()
        self.agents: Dict[str, Callable[[str, int, int, Callable], BaseMLAgent]] = {
            "explore": lambda env_name, s, a, log: A3CAgent(
                env_name, s, a, learning_rate=self.learning_rate, gamma=self.gamma, log_print=log
            ),
            "random_learner": lambda env_name, s, a, log: RandomA3CAgent(
                env_name, s, a, learning_rate=self.learning_rate, gamma=self.gamma, log_print=log
            ),
            "scripted": lambda env_name, s, a, log: SemiRandomA3CAgent(
                env_name, s, a, learning_rate=self.learning_rate, gamma=self.gamma, log_print=log
            ),
            "learning": lambda env_name, s, a, log: A3CAgent(
                env_name,
                s,
                a,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                log_print=log,
                temperature_episodes=0,
            ),
            "examplemask": lambda env_name, s, a, log: A3CAgent(
                env_name, s, a, log_print=log, temperature_episodes=0, mask=[(10, 0), (2, 0.005), (3, 0.05)]
            ),
            "play": lambda env_name, s, a, log: PlayA3CAgent(env_name, s, a, log_print=log, temperature_episodes=0),
            "optimal": lambda env_name, s, a, log: ArgMaxA3CAgent(
                env_name,
                s,
                a,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                log_print=log,
                temperature_episodes=0,
            ),
            "random": lambda env_name, s, a, log: RandomAgent(s, a),
            "scriptonly": lambda env_name, s, a, log: SemiScriptedAgent(s, a),
        }

        self.action = 0
        self.agent_dummy = agent == "scripted" or agent == "scriptonly"
        self.agent_needs_state = agent != "random" and agent != "scriptonly"
        self.learning_rate = 0.005
        self.gamma = 0.995
        self.score = 0
        self.build = build
        self.build_name = build_str
        self.agent_name = agent

    def choose_action(self, state: Optional["ndarray"]) -> int:
        """
        Choose and return the next action.
        :param state: numpy array
        :return: action type integer
        """
        self.action = self.agent.choose_action(state, self.score)
        return self.action

    async def start(self, knowledge: "Knowledge"):
        await super().start(knowledge)
        self.agent = self.agents[self.agent_name](
            self.build_name, self.build.state_size, self.build.action_size, knowledge.print
        )
        self.build.agent = self.agent
        await self.build.start(knowledge)

    async def update(self):
        state = None
        if self.agent_needs_state:
            state = np.array(self.build.state)

        self.score = self.build.score

        if self.agent_dummy:
            self.agent.action = self.scripted_action()

        self.build.action = self.choose_action(state)
        await self.build.execute()

    @abstractmethod
    def scripted_action(self) -> int:
        pass

    async def post_update(self):
        if self.debug:
            action_name, color = self.build.get_action_name_color(self.action)
            self.ai.client.debug_text_screen(action_name, (0.01, 0.01), color, 16)
            self.ai.client.debug_text_screen(str(self.score), (0.00, 0.05), color, 16)

    async def on_end(self, game_result: Result):
        self.build.on_end(game_result)
