from sc2 import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.knowledges import KnowledgeBot
from sharpy.managers.roles import UnitTask
from sharpy.plans import BuildOrder, StepBuildGas, SequentialList, Step
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import AutoOverLord, MorphLair, ZergUnit
from sharpy.plans.require import RequiredGas, RequireCustom, RequiredUnitExists, RequiredAny, RequiredTechReady
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import InjectLarva


class WorkerRushDefender(KnowledgeBot):
    """Dummy bot for debugging ML learning."""

    def __init__(self):
        super().__init__(type(self).__name__)

    async def create_plan(self) -> BuildOrder:
        return BuildOrder([
            SequentialList([
                ActUnit(UnitTypeId.DRONE, UnitTypeId.LARVA, 14),
                ActExpand(2),
                ActBuilding(UnitTypeId.SPAWNINGPOOL, 1),
                ActUnit(UnitTypeId.OVERLORD, UnitTypeId.LARVA, 2),
                ActUnit(UnitTypeId.QUEEN, UnitTypeId.HATCHERY, 1),
                ActUnit(UnitTypeId.ZERGLING, UnitTypeId.LARVA, 200),
            ]),
            SequentialList(
                [
                    PlanDistributeWorkers(),
                    PlanZoneDefense(),
                    AutoOverLord(),
                    InjectLarva(),
                    PlanZoneGather(),
                    PlanZoneAttack(10),
                    PlanFinishEnemy(),
                ])
        ])

    async def on_step(self, iteration):
        await super().on_step(iteration)
        if len(self.townhalls) > 0 and self.enemy_units.closer_than(10, self.townhalls[0].position):
            for worker in self.workers:
                self.knowledge.roles.set_task(UnitTask.Attacking, worker)
        else:
            for worker in self.workers:
                self.knowledge.roles.set_task(UnitTask.Gathering, worker)

