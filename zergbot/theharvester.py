from typing import Dict, Callable, Union, List

from sc2 import Result
from sharpy.knowledges import KnowledgeBot
from sharpy.managers.roles import UnitTask
from sharpy.plans import BuildOrder
from zergbot.builds import *
from zergbot.builds.worker_distraction import WorkerDistraction_v0
from zergbot.ml.agents import *
from a3c_general_agent.a3c_sc2_migrate import A3CAgent

agents: Dict[str, Callable[[int, int], BaseMLAgent]] = {
    "learning": lambda s, a: A3CAgent(s, a),
    "random": lambda s, a: RandomAgent(s, a),
    "scripted": lambda s, a: SemiScriptedAgent(s, a)
}

builds: Dict[str, Callable[[], MlBuild]] = {
    "default": lambda: EconLings_v0(),
    "workerdistraction": lambda: WorkerDistraction_v0()
}


class HarvesterBot(KnowledgeBot):
    agent: BaseMLAgent
    ml_build: MlBuild


    def __init__(self, agent: str = "random", build: str = "default"):
        super().__init__("Harvester")
        if build not in builds:
            raise ValueError(f'{build} does not exist')
        self.build_text = build
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.initialize_agent(agent, build)

    def initialize_agent(self, agent: str, build_text):
        self.ml_build = builds[build_text]()

        if isinstance(agent, BaseMLAgent):
            self.agent = agent
        else:
            self.agent = agents[agent](self.ml_build.state_size, self.ml_build.action_size)

        self.ml_build.agent = self.agent

    async def create_plan(self) -> BuildOrder:
        self.knowledge.data_manager.set_build(self.build_text)
        self.knowledge.print(self.build_text, "Build")

        return self.ml_build

    async def on_end(self, game_result: Result):
        self.ml_build.on_end(game_result)
        await super().on_end(game_result)



