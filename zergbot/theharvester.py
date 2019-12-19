from typing import Dict, Callable, Union

from sc2 import UnitTypeId, Result
from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from zergbot.builds import *
from zergbot.ml.agents import *

agents: Dict[str, Callable[[int, int], BaseMLAgent]] = {
    "learning": lambda s, a: A3CAgent(s, a),
    "random": lambda s, a: RandomAgent(s, a),
    "scripted": lambda s, a: SemiScriptedAgent(s, a)
}


builds: Dict[str, Callable[[], MlBuild]] = {
    "default": lambda: EconLings_v0()
}


class HarvesterBot(KnowledgeBot):
    agent: BaseMLAgent
    ml_build: MlBuild
    def __init__(self, agent: Union[str, BaseMLAgent] = "random", build: str = "default"):
        super().__init__("Harvester")
        self.agent = agent
        if build not in agents:
            raise ValueError(f'{build} does not exist')
        self.build_text = build
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.next_action = 0
        self.initialize_agent(agent, build)

    def initialize_agent(self, agent: Union[str, BaseMLAgent], build_text):
        self.ml_build = builds[build_text]()

        if isinstance(agent, BaseMLAgent):
            self.agent = agent
        else:
            self.agent = agents[self.build_text](self.ml_build.state_size, self.ml_build.action_size)

    async def create_plan(self) -> BuildOrder:
        self.knowledge.data_manager.set_build(self.build_text)
        self.knowledge.print(self.build_text, "Build")

        economy = Step(lambda k: self.next_action == 0, SequentialList([
            ZergUnit(UnitTypeId.DRONE, 15),
            ActExpand(2),
            ZergUnit(UnitTypeId.DRONE, 24),
            ActExpand(3),
            ZergUnit(UnitTypeId.DRONE, 38),
            ActExpand(4),
            ZergUnit(UnitTypeId.DRONE, 50),
        ]))

        units = Step(lambda k: self.next_action == 1, SequentialList([
            ActBuilding(UnitTypeId.SPAWNINGPOOL),
            BuildOrder([
                Step(RequiredMinerals(500), ActExpand(4)),
                Step(None, ZergUnit(UnitTypeId.QUEEN, 5), skip_until=lambda k: self.minerals > 150),
                ZergUnit(UnitTypeId.ZERGLING, 400),
            ])
        ]))

        tactics = SequentialList([
            PlanDistributeWorkers(),
            WorkerScout(),
            SpreadCreep(),
            InjectLarva(),
            PlanHeatOverseer(),

            PlanWorkerOnlyDefense(),
            PlanZoneDefense(),
            PlanZoneGather(),
            PlanZoneAttack(),
            PlanFinishEnemy(),
        ])

        return BuildOrder([
            AutoOverLord(),
            economy,
            units,
            tactics
        ])

    async def on_step(self, iteration):
        self.next_action = self.agent.choose_action([self.time, self.supply_workers, self.supply_army])

        # todo: turn off for ladder.
        if self.next_action == 0:
            self.client.debug_text_screen("ECON", (0.01, 0.01), (0, 255, 0), 16)
        else:
            self.client.debug_text_screen("ARMY", (0.01, 0.01), (255, 0, 0), 16)

        if not self.conceded and self.knowledge.game_analyzer.bean_predicting_defeat_for > 5:
            await self.chat_send("pineapple")
            self.conceded = True
        return await super().on_step(iteration)

    async def on_end(self, game_result: Result):
        self.agent.on_end(game_result)
        await super().on_end(game_result)
