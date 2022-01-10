from abc import abstractmethod, ABC
from typing import Dict, Callable, Optional
import numpy as np
from sc2 import Result
from sc2.position import Point2
from sharpy.managers import ManagerBase
from sharpy.managers.extensions import ChatManager
from tactics.ml.agents import *
from tactics.ml.ml_build import MlBuild


class BaseAgentManager(ManagerBase):
    agent: BaseMLAgent
    build: MlBuild
    chatter: ChatManager

    def __init__(self, agent: str, build_str: str, build: MlBuild, shared_global_vars: dict,
                 learning_rate=0.001, update_freq=-1, model_file_lock_timeout=999999, gamma=0.99) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.model_file_lock_timeout = model_file_lock_timeout
        self.gamma = gamma

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
                update_freq=self.update_freq,
                gamma=self.gamma,
                model_file_lock_timeout=self.model_file_lock_timeout,
                log_print=log,
                temperature_episodes=0,
                shared_global_vars=self.shared_global_vars,
            ),
            "datalearning": lambda env_name, s, a, log: RecordLearner(
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

        self.start_time = 0  # in seconds
        self.action = 0
        self.agent_dummy = agent == "scripted" or agent == "scriptonly"
        self.agent_needs_state = agent != "random" and agent != "scriptonly"
        self.score = 0
        self.build = build
        self.build_name = build_str
        self.agent_name = agent
        self.shared_global_vars = shared_global_vars

    def choose_action(self, state: Optional["ndarray"]) -> int:
        """
        Choose and return the next action.
        :param state: numpy array
        :return: action type integer
        """
        old_action = self.action
        if self.build.ready_to_update:
            self.action = self.agent.choose_action(state, self.score)

        if self.action != old_action:
            self.print(f"Agent {self.agent_name} changed action to {self.action}")

        return self.action

    async def start(self, knowledge: "Knowledge"):
        await super().start(knowledge)
        self.chatter: ChatManager = self.knowledge.get_required_manager(ChatManager)
        self.agent = self.create_agent(knowledge.print)
        await self.build.start(knowledge)

    def create_agent(self, log: Callable[[str], None]):
        agent = self.agents[self.agent_name](self.build_name, self.build.state_size, self.build.action_size, log)
        self.build.agent = agent
        return agent

    async def chat_space(self):
        if self.ai.time > 12:
            await self.chatter.chat_taunt_once(
                f"{self.key}_ml_state_space", lambda: f"State size {self.agent.state_size}"
            )
        if self.ai.time > 20:
            await self.chatter.chat_taunt_once(
                f"{self.key}_ml_action_space", lambda: f"Action size {self.agent.action_size}"
            )
        if self.ai.time > 25:
            await self.chatter.chat_taunt_once(f"{self.key}_ml_actions", self.build.write_actions)
        if self.ai.time > 40:
            if self.agent_name == "scriptonly":
                await self.chatter.chat_taunt_once(
                    f"{self.key}_ml_episodes", lambda: f"Learning is disabled and this agent uses script only"
                )
            elif self.agent_name == "scripted":
                await self.chatter.chat_taunt_once(
                    f"{self.key}_ml_episodes",
                    lambda: f"This agent uses script only but has trained for {self.agent.episode} episodes",
                )
            elif self.agent_name == "random_learner":
                await self.chatter.chat_taunt_once(
                    f"{self.key}_ml_episodes",
                    lambda: f"This agent is random but has trained for {self.agent.episode} episodes",
                )
            else:
                await self.chatter.chat_taunt_once(
                    f"{self.key}_ml_episodes", lambda: f"This agent has trained for {self.agent.episode} episodes"
                )

    async def update(self):
        if self.ai.time > self.start_time:
            state = None
            if self.agent_needs_state:
                state = np.array(self.build.state)
            self.build.calc_reward()
            self.score = self.build.score

            if self.agent_dummy:
                self.agent.action = self.scripted_action()

            self.build.action = self.choose_action(state)
        else:
            self.build.action = self.action
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
