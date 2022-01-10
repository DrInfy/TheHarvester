from typing import List, Union, Tuple

from sc2 import UnitTypeId
from sharpy.plans import BuildOrder
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import Minerals
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from tactics.ml.ml_build import MlBuild


class EconLings_v0(MlBuild):
    STATE_SIZE = 8
    ACTION_SIZE = 2

    def __init__(self):
        super().__init__(EconLings_v0.STATE_SIZE, EconLings_v0.ACTION_SIZE, self.create_plan(), result_multiplier=10000)

    @property
    def state(self) -> List[Union[int, float]]:
        return [
            self.ai.time,
            self.ai.supply_workers,
            self.ai.supply_army,
            self.ai.minerals,
            len(self.cache.own(UnitTypeId.LARVA)),
            self.get_count(UnitTypeId.DRONE),
            self.get_count(UnitTypeId.ZERGLING),
            self.get_count(UnitTypeId.HATCHERY),
        ]

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        if action == 0:
            return ("ECON", (0, 255, 0))
        return ("ARMY", (255, 0, 0))

    @property
    def score(self) -> float:
        self.reward = 0
        self.reward -= min(0, self.ai.minerals - 400) / 1000.0
        self.reward += self.ai.income_calculator.mineral_income / 100
        # return self.reward
        value = self.ai.game_analyzer.enemy_predicter.own_army_value_minerals + \
            self.ai.game_analyzer.enemy_predicter.own_army_value_gas
        return super().score - min(0, self.ai.minerals - 400) / 1000.0 + value / 1000.0

    def create_plan(self) -> List[Union[ActBase, List[ActBase]]]:
        economy = Step(
            lambda k: self.action == 0,
            SequentialList(
                [
                    ZergUnit(UnitTypeId.DRONE, 15),
                    Expand(2),
                    ZergUnit(UnitTypeId.DRONE, 24),
                    Expand(3),
                    ZergUnit(UnitTypeId.DRONE, 38),
                    Expand(4),
                    ZergUnit(UnitTypeId.DRONE, 50),
                    Expand(5),
                ]
            ),
        )

        units = Step(
            lambda k: self.action == 1,
            SequentialList(
                [
                    ActBuilding(UnitTypeId.SPAWNINGPOOL),
                    BuildOrder(
                        [
                            # Step(RequiredMinerals(500), ActExpand(4)),
                            Step(None, ZergUnit(UnitTypeId.QUEEN, 5), skip_until=Minerals(150)),
                            ZergUnit(UnitTypeId.ZERGLING, 400),
                        ]
                    ),
                ]
            ),
        )

        tactics = SequentialList(
            [
                DistributeWorkers(),
                WorkerScout(),
                SpreadCreep(),
                InjectLarva(),
                PlanHeatOverseer(),
                PlanWorkerOnlyDefense(),
                PlanZoneDefense(),
                PlanZoneGather(),
                PlanZoneAttack(),
                PlanFinishEnemy(),
            ]
        )

        return [AutoOverLord(), economy, units, tactics]
