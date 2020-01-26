from typing import List, Union, Tuple

from sc2 import UnitTypeId
from sharpy.plans import BuildOrder
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from zergbot.builds.ml_build import MlBuild
from zergbot.ml.agents import BaseMLAgent


class EconLings_v0(MlBuild):

    def __init__(self, agent: BaseMLAgent):
        super().__init__(agent, 4, 2, self.create_plan())

    def state(self) -> List[Union[int, float]]:
        return [self.ai.time, self.ai.supply_workers, self.ai.supply_army, self.ai.minerals]

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        if self.next_action == 0:
            return ("ECON", (0, 255, 0))
        return ("ARMY", (255, 0, 0))

    async def execute(self) -> bool:
        self.next_action = self.agent.choose_action([self.time, self.supply_workers, self.supply_army])

        # todo: turn off for ladder.
        if self.next_action == 0:
            self.client.debug_text_screen("ECON", (0.01, 0.01), (0, 255, 0), 16)
        else:
            self.client.debug_text_screen("ARMY", (0.01, 0.01), (255, 0, 0), 16)
        return await super().execute()

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
