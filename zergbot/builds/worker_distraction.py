from abc import abstractmethod
from typing import List, Union

from sc2 import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.plans import BuildOrder, StepBuildGas
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from zergbot.builds.ml_build import MlBuild


class WorkerDistraction_v0(MlBuild):

    def __init__(self):
        super().__init__(4, 2, self.create_plan())

    @property
    def state(self) -> List[int]:
        try:
            distraction_worker = self.ai.workers.by_tag(self.ai.distraction_worker_tag)
            return [len(self.ai.enemy_units.closer_than(2, distraction_worker.position)) > 1, True]
        except KeyError:
            return [False, False]

    def create_plan(self) -> List[Union[ActBase, List[ActBase]]]:
        return [
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
        ]
