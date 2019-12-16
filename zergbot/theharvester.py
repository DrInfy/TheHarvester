from math import floor

from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from sharpy.plans.tactics.zone_attack import AttackStatus
from sc2 import UnitTypeId, Race
from sc2.ids.upgrade_id import UpgradeId

from sharpy.plans import BuildOrder, Step, SequentialList, StepBuildGas
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *

from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder
from zergbot.ml.agents import RandomAgent, SemiScriptedAgent, A3CAgent


class HarvesterBot(KnowledgeBot):
    def __init__(self, build: str = "default"):
        super().__init__("Harvester")
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.next_action = 0

        # todo: make this selection more elegant
        # self.agent = RandomAgent()
        self.agent = SemiScriptedAgent()
        # self.agent = A3CAgent()

    async def create_plan(self) -> BuildOrder:
        self.knowledge.data_manager.set_build("self learning")
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
        self.next_action = self.agent.choose_action((self.time, self.supply_workers, self.supply_army))

        if not self.conceded and self.knowledge.game_analyzer.bean_predicting_defeat_for > 5:
            await self.chat_send("pineapple")
            self.conceded = True
        return await super().on_step(iteration)
