import string
from typing import Dict, Callable, Union, List, Optional

from sc2 import Result
from sharpy.knowledges import KnowledgeBot
from sharpy.managers.roles import UnitTask
from sharpy.plans import BuildOrder
from harvester.builds import *
from harvester.ml.agents import *

agents: Dict[str, Callable[[int, int], BaseMLAgent]] = {

    "explore": lambda env_name, s, a, log: A3CAgent(env_name, s, a, log_print=log),
    "random_learner": lambda env_name, s, a, log: RandomA3CAgent(env_name, s, a, log_print=log),
    "scripted2": lambda env_name, s, a, log: SemiRandomA3CAgent(env_name, s, a, log_print=log),
    "learning": lambda env_name, s, a, log: A3CAgent(env_name, s, a, log_print=log, logit_bonus_episodes=0),
    "play": lambda env_name, s, a, log: PlayA3CAgent(env_name, s, a, log_print=log, logit_bonus_episodes=0),
    "optimal": lambda env_name, s, a, log: ArgMaxA3CAgent(env_name, s, a, log_print=log, logit_bonus_episodes=0),
    "random": lambda env_name, s, a, log: RandomAgent(s, a),
    "scripted": lambda env_name, s, a, log: SemiScriptedAgent(s, a),
}

builds: Dict[str, Callable[[], MlBuild]] = {
    "default": lambda: AllBuild(),
    "econ_lings": lambda: EconLings_v0(),
    "roach": lambda: EconLingRoach(),
}


class HarvesterBot(KnowledgeBot):
    agent: BaseMLAgent
    ml_build: MlBuild


    def __init__(self, agent: str, build: str, model_index: Optional[str] = None):
        super().__init__("Harvester")
        if build not in builds:
            raise ValueError(f'{build} does not exist')
        self.build_text = build + "." + agent
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.agent_str = agent
        self.build_str = build
        self.model_index = model_index


    def initialize_agent(self, agent: str, build: str):
        self.ml_build = builds[build]()

        if isinstance(agent, BaseMLAgent):
            self.agent = agent
        else:
            if self.model_index:
                build += "." + self.model_index
            self.agent = agents[agent](build, self.ml_build.state_size, self.ml_build.action_size, self.knowledge.print)

        self.ml_build.agent = self.agent

    async def create_plan(self) -> BuildOrder:
        if self.model_index:
            self.build_text += "." + self.model_index

        self.initialize_agent(self.agent_str, self.build_str)
        self.knowledge.data_manager.set_build(self.build_text)
        self.knowledge.print(self.build_text, "Build")

        return self.ml_build

    async def on_end(self, game_result: Result):
        self.ml_build.on_end(game_result)
        await super().on_end(game_result)



