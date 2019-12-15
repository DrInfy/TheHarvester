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


class HarvesterBot(KnowledgeBot):
    def __init__(self, build: str = "default"):
        super().__init__("Harvester")
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.next_action = 0

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
            ZergUnit(UnitTypeId.ZERGLING, 50),
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
        if not self.conceded and self.knowledge.game_analyzer.bean_predicting_defeat_for > 5:
            await self.chat_send("pineapple")
            self.conceded = True
        return await super().on_step(iteration)
