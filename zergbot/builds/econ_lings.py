from typing import List, Union

from sc2 import UnitTypeId
from sharpy.plans import BuildOrder
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from zergbot.builds.ml_build import MlBuild


class EconLings_v0(MlBuild):

    def __init__(self):
        super().__init__(4, 2, self.create_plan())

    def state(self) -> List[int]:
        return [self.ai.time, self.ai.supply_workers, self.ai.supply_army, self.ai.minerals]

    def create_plan(self) -> List[Union[ActBase, List[ActBase]]]:
        economy = Step(lambda k: self.action == 0, SequentialList([
            ZergUnit(UnitTypeId.DRONE, 15),
            ActExpand(2),
            ZergUnit(UnitTypeId.DRONE, 24),
            ActExpand(3),
            ZergUnit(UnitTypeId.DRONE, 38),
            ActExpand(4),
            ZergUnit(UnitTypeId.DRONE, 50),
        ]))

        units = Step(lambda k: self.action == 1, SequentialList([
            ActBuilding(UnitTypeId.SPAWNINGPOOL),
            BuildOrder([
                Step(RequiredMinerals(500), ActExpand(4)),
                Step(None, ZergUnit(UnitTypeId.QUEEN, 5), skip_until=lambda k: k.ai.minerals > 150),
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

        return [
            AutoOverLord(),
            economy,
            units,
            tactics
        ]