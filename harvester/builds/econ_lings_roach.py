from typing import List, Union, Tuple

from sc2 import UnitTypeId
from sharpy.general.extended_power import ExtendedPower
from sharpy.plans import BuildOrder
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.require.supply import SupplyType
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from tactics.ml.ml_build import MlBuild
from tactics import *


class EconLingRoach(MlBuild):
    def __init__(self):
        super().__init__(25, 6, self.create_plan(), result_multiplier=1000)

    @property
    def state(self) -> List[Union[int, float]]:
        my_lost_m, my_lost_g = self.lost_units_manager.calculate_own_lost_resources()
        enemy_lost_m, enemy_lost_g = self.lost_units_manager.calculate_enemy_lost_resources()
        enemy_power: ExtendedPower = self.enemy_army_predicter.predicted_enemy_power
        return [
            self.ai.time,
            self.ai.supply_workers,
            self.ai.supply_army,
            self.ai.minerals,
            len(self.cache.own(UnitTypeId.LARVA)),  # 5
            self.get_count(UnitTypeId.DRONE),
            self.get_count(UnitTypeId.ZERGLING),
            self.get_count(UnitTypeId.QUEEN),
            self.get_count(UnitTypeId.ROACH),
            self.get_ml_number(UnitTypeId.HATCHERY),  # 10
            self.get_ml_number(UnitTypeId.SPAWNINGPOOL),
            self.get_ml_number(UnitTypeId.ROACHWARREN),
            self.get_ml_number(UnitTypeId.EXTRACTOR),
            self.ai.enemy_army_predicter.enemy_known_worker_count,
            self.ai.enemy_army_predicter.enemy_mined_minerals,  # 15
            self.ai.enemy_army_predicter.enemy_army_known_minerals,
            enemy_power.power,
            enemy_power.air_presence,
            enemy_power.melee_power,
            self.ai.vespene,  # 20
            my_lost_m,
            my_lost_g,
            enemy_lost_m,
            enemy_lost_g,
            self.zone_manager.expansion_zones[0].paths[-1].distance,  # 25
        ]

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        if action == 0:
            return ("ECON bases", (0, 255, 0))
        if action == 1:
            return ("ECON drones", (0, 255, 0))
        if action == 2:
            return ("ECON queens", (0, 255, 0))
        if action == 3:
            return ("ARMY lings", (255, 0, 0))
        if action == 4:
            return ("ARMY roaches", (255, 0, 0))
        if action == 5:
            return ("ARMY hydras", (255, 0, 0))
        return ("UNKNOWN " + str(action), (255, 255, 255))

    async def chat_space(self):
        await super().chat_space()
        if self.ai.time > 30:
            await self.chatter.chat_taunt_once(
                "actions", lambda: f"Actions: 0: bases, 1: drones, 2: queens, 3: lings, 4:roaches"
            )

    @property
    def score(self) -> float:
        self.reward = 0
        my_lost_m, my_lost_g = self.lost_units_manager.calculate_own_lost_resources()
        enemy_lost_m, enemy_lost_g = self.lost_units_manager.calculate_enemy_lost_resources()
        my_lost = my_lost_m + my_lost_g
        enemy_lost = enemy_lost_m + enemy_lost_g
        sum_lost = my_lost + enemy_lost
        if sum_lost > 0:
            self.reward += (enemy_lost / sum_lost - 0.5) * 0.5 * min(sum_lost, 1000)
        self.reward += min(100, self.ai.state.score.collection_rate_minerals / 20)
        self.reward += min(100, self.ai.state.score.collection_rate_vespene / 20)
        self.reward -= min(200, max(0, self.ai.minerals - 500) / 40.0)
        self.reward -= min(200, max(0, self.ai.vespene - 300) / 40.0)
        return self.reward

    def create_plan(self) -> List[Union[ActBase, List[ActBase]]]:
        pool = PositionBuilding(UnitTypeId.SPAWNINGPOOL, DefensePosition.BehindMineralLineLeft, 0)
        bases = Step(lambda k: self.action == 0, Expand(10))
        drones = Step(lambda k: self.action == 1, ZergUnit(UnitTypeId.DRONE))

        queens = Step(lambda k: self.action == 2, SequentialList([pool, BuildOrder([ZergUnit(UnitTypeId.QUEEN),])]))

        lings = Step(lambda k: self.action == 3, SequentialList([pool, ZergUnit(UnitTypeId.ZERGLING),]))

        roaches = Step(
            lambda k: self.action == 4,
            SequentialList(
                [
                    pool,
                    Step(
                        UnitReady(UnitTypeId.SPAWNINGPOOL),
                        PositionBuilding(UnitTypeId.ROACHWARREN, DefensePosition.BehindMineralLineRight, 0),
                    ),
                    BuildGas(1),
                    Step(None, BuildGas(2), skip_until=Supply(20, supply_type=SupplyType.Workers)),
                    Step(None, BuildGas(3), skip=Gas(300), skip_until=Supply(30, supply_type=SupplyType.Workers),),
                    ZergUnit(UnitTypeId.ROACH),
                ]
            ),
        )

        hydras = Step(
            lambda k: self.action == 5,
            SequentialList(
                [
                    pool,
                    Step(
                        UnitReady(UnitTypeId.SPAWNINGPOOL),
                        PositionBuilding(UnitTypeId.ROACHWARREN, DefensePosition.BehindMineralLineRight, 0),
                    ),
                    BuildGas(2),
                    MorphLair(),
                    ActBuilding(UnitTypeId.HYDRALISKDEN, 1),
                    Step(None, BuildGas(3), skip_until=Supply(30, supply_type=SupplyType.Workers)),
                    Step(None, BuildGas(4), skip=Gas(300), skip_until=Supply(30, supply_type=SupplyType.Workers),),
                    ZergUnit(UnitTypeId.HYDRALISK),
                ]
            ),
        )

        tactics = SequentialList(
            [
                DistributeWorkers(1),
                PlanGatherOptimizer(),
                OverlordScout(),
                Step(None, WorkerScout(), skip_until=Time(70)),
                SpreadCreep(),
                InjectLarva(),
                PlanHeatOverseer(),
                PlanWorkerOnlyDefense(),
                PlanZoneDefense(),
                PlanZoneGather(),
                PlanZoneAttack(15),
                PlanFinishEnemy(),
            ]
        )

        return [
            [
                ZergUnit(UnitTypeId.OVERLORD, 1),
                Step(None, ZergUnit(UnitTypeId.OVERLORD, 2), skip_until=Supply(13)),
                AutoOverLord(),
            ],
            bases,
            queens,
            drones,
            lings,
            roaches,
            hydras,
            tactics,
        ]
