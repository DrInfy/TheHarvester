from typing import Dict, Callable, Optional, List

from harvester.agent_manager import ZergAgentManager
from managers import ExtendedUnitManager
from sc2 import Result
from sharpy.knowledges import KnowledgeBot
from sharpy.managers.extensions import BuildDetector, ChatManager
from sharpy.plans import BuildOrder
from harvester.builds import *
from tactics.ml.agents import *
from tactics.ml.ml_build import MlBuild


builds: Dict[str, Callable[[], MlBuild]] = {
    "default": lambda: AllBuild(),
    "econ_lings": lambda: EconLings_v0(),
    "roach": lambda: EconLingRoach(),
}


class HarvesterBot(KnowledgeBot):
    agent: BaseMLAgent
    ml_build: MlBuild

    def __init__(self, agent: str, build: str, model_index: Optional[str] = None):
        super().__init__("HarvesterZerg")
        if build not in builds:
            raise ValueError(f"{build} does not exist")
        self.build_text = build + "." + agent
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.agent_str = agent
        self.build_str = build
        self.model_index = model_index

    def configure_managers(self) -> Optional[List["ManagerBase"]]:
        self.knowledge.log_manager.start_with = "[EDGE] "
        build_detector = BuildDetector()

        agent_manager = self.create_agent_manager()
        return [build_detector, ExtendedUnitManager(), ChatManager(), agent_manager]

    def create_agent_manager(self):
        if self.model_index:
            self.build_text += "." + self.model_index
            model_name = self.build_str + "." + self.model_index
        else:
            model_name = self.build_str
        agent_manager = ZergAgentManager(self.agent_str, model_name, builds[self.build_str]())
        return agent_manager

    async def create_plan(self) -> "BuildOrder":
        return BuildOrder([])

    async def on_start(self):
        await super().on_start()
        self.knowledge.data_manager.set_build(self.build_text)
        self.knowledge.print(self.build_text, "Build")
