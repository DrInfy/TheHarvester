from typing import List, Union, Tuple, Callable

from sc2 import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.general.extended_power import ExtendedPower
from sharpy.general.zone import Zone
from sharpy.managers.build_detector import EnemyRushBuild
from sharpy.plans import BuildOrder, StepBuildGas
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.require.required_supply import SupplyType
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from harvester.builds.ml_build import MlBuild
# from tactics import *
from sharpy.managers import BuildDetector
from sharpy.managers.enemy_units_manager import EnemyUnitsManager
from harvester.builds.units import *

class AllBuild(MlBuild):
    eu: EnemyUnitsManager
    bp: BuildDetector

    def __init__(self):
        self.statesf: List[Callable[[], float]] = []
        self.statesr: List[float] = []
        self.attack = PlanZoneAttack(15)
        orders = self.create_builds()
        self.init_funcs()
        super().__init__(len(self.statesf), 7, orders, result_multiplier=1000)
        self.minimum_action_time = 5
        self.update_on_mineral_loss = False

    def init_funcs(self):
        self.statesf.append(lambda: self.ai.time)
        self.statesf.append(lambda: self.ai.minerals)
        self.statesf.append(lambda: self.ai.vespene)
        self.statesf.append(lambda: self.ai.supply_workers)
        self.statesf.append(lambda: self.ai.supply_army)
        self.statesf.append(lambda: self.knowledge.rush_distance)
        self.statesf.append(lambda: self.ai.state.score.collection_rate_minerals)
        self.statesf.append(lambda: self.ai.state.score.collection_rate_vespene)
        self.statesf.append(lambda: self.ai.state.score.killed_minerals_army)
        self.statesf.append(lambda: self.ai.state.score.killed_vespene_army)

        for unit_type in zerg_units:
            self.statesf.append(lambda tmp=unit_type: self.get_count(tmp))

        for building_type in zerg_buildings:
            self.statesf.append(lambda tmp=building_type: self.get_ml_number(tmp))

        self.add_enemy_zerg()
        self.add_enemy_terran()
        self.add_enemy_protoss()

    def add_enemy_zerg(self):
        for unit_type in zerg_units:
            self.statesf.append(lambda: self.eu.unit_count(unit_type))

        for building_type in zerg_buildings:
            self.statesf.append(lambda: self.get_building_timing(building_type, 0))

        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.HATCHERY, 1))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.HATCHERY, 2))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.HATCHERY, 3))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.HATCHERY, 4))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.HATCHERY, 5))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.HATCHERY, 6))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.EVOLUTIONCHAMBER, 1))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.SPIRE, 1))

    def add_enemy_terran(self):
        for unit_type in terran_units:
            self.statesf.append(lambda: self.eu.unit_count(unit_type))

        for building_type in terran_buildings:
            self.statesf.append(lambda: self.get_building_timing(building_type, 0))

        self.add_timings(UnitTypeId.COMMANDCENTER, 6)
        self.add_timings(UnitTypeId.BARRACKS, 6)
        self.add_timings(UnitTypeId.FACTORY, 4)
        self.add_timings(UnitTypeId.STARPORT, 3)

    def add_enemy_protoss(self):
        for unit_type in protoss_units:
            self.statesf.append(lambda tmp=unit_type: self.eu.unit_count(tmp))

        for building_type in protoss_buildings:
            self.statesf.append(lambda tmp=building_type: self.get_building_timing(tmp, 0))

        self.add_timings(UnitTypeId.NEXUS, 6)
        self.add_timings(UnitTypeId.GATEWAY, 8)
        self.add_timings(UnitTypeId.ROBOTICSFACILITY, 3)
        self.add_timings(UnitTypeId.STARPORT, 3)
        self.add_timings(UnitTypeId.PYLON, 6)
        self.add_timings(UnitTypeId.PHOTONCANNON, 5)

    async def start(self, knowledge: 'Knowledge'):
        await super().start(knowledge)
        self.eu: EnemyUnitsManager = self.knowledge.enemy_units_manager
        self.bp: BuildDetector = self.knowledge.build_detector

    def add_timings(self, unit_type: UnitTypeId, count: int):
        for index in range(1, count):
            self.statesf.append(lambda: self.get_building_timing(unit_type, index))

    def get_building_timing(self, unit_type: UnitTypeId, index: int) -> float:
        timing_list = self.bp.timings.get(unit_type, None)
        if timing_list is None or len(timing_list) <= index:
            return 0
        return timing_list[index]

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
        if action == 6:
            return ("ARMY banelings", (255, 0, 0))
        return ("UNKNOWN " + str(action), (255, 255, 255))

    @property
    def state(self) -> List[Union[int, float]]:
        self.statesr.clear()
        for func in self.statesf:
            self.statesr.append(func())
        return self.statesr

    def create_builds(self) -> List[Union[ActBase, List[ActBase]]]:
        tactics = SequentialList([
            # PlanGatherOptimizer(),
            OverlordScout(),
            # Step(None, LingScoutMain(), skip_until=RequiredTime(4 * 60)),
            # Step(None, LingScoutMain(), skip_until=RequiredTime(8 * 60)),
            Step(None, WorkerScout(), skip_until=RequiredTime(40)),
            SpreadCreep(),
            InjectLarva(),
            PlanHeatOverseer(),

            PlanWorkerOnlyDefense(),
            PlanZoneDefense(),
            PlanZoneGather(),
            self.attack,
            PlanFinishEnemy(),
        ])
        gas = SequentialList([
            Step(None, StepBuildGas(2), skip=RequiredGas(200),
                 skip_until=RequiredSupply(25, supply_type=SupplyType.Workers)),
            Step(None, StepBuildGas(3), skip=RequiredGas(200),
                 skip_until=RequiredSupply(40, supply_type=SupplyType.Workers)),
            Step(None, StepBuildGas(4), skip=RequiredGas(200),
                 skip_until=RequiredSupply(50, supply_type=SupplyType.Workers)),
        ])
        pool = PositionBuilding(UnitTypeId.SPAWNINGPOOL, DefensePosition.BehindMineralLineLeft, 0)
        bases = Step(lambda k: self.action == 0, ActExpand(10))
        drones = Step(lambda k: self.action == 1, ZergUnit(UnitTypeId.DRONE, 100))

        queens = Step(lambda k: self.action == 2, SequentialList([
            pool,
            BuildOrder([
                ZergUnit(UnitTypeId.QUEEN),
            ])
        ]))

        lings = Step(lambda k: self.action == 3, SequentialList([
            pool,
            ZergUnit(UnitTypeId.ZERGLING),
        ]))

        banelings = Step(lambda k: self.action == 6, SequentialList([
            pool,
            StepBuildGas(1),
            Step(RequiredUnitReady(UnitTypeId.SPAWNINGPOOL),ActBuilding(UnitTypeId.BANELINGNEST)),
            gas,
            ZergUnit(UnitTypeId.BANELING),
        ]))

        roaches = Step(lambda k: self.action == 4, SequentialList([
            pool,
            Step(RequiredUnitReady(UnitTypeId.SPAWNINGPOOL),
                 PositionBuilding(UnitTypeId.ROACHWARREN, DefensePosition.BehindMineralLineRight, 0)),
            StepBuildGas(1),
            gas,
            ZergUnit(UnitTypeId.ROACH),
        ]))

        hydras = Step(lambda k: self.action == 5, SequentialList([
            pool,
            Step(RequiredUnitReady(UnitTypeId.SPAWNINGPOOL),
                 PositionBuilding(UnitTypeId.ROACHWARREN, DefensePosition.BehindMineralLineRight, 0)),
            StepBuildGas(1),
            MorphLair(),
            ActBuilding(UnitTypeId.HYDRALISKDEN, 1),
            gas,
            ZergUnit(UnitTypeId.HYDRALISK),
        ]))

        return [
            [
                ZergUnit(UnitTypeId.OVERLORD, 1),
                Step(None, ZergUnit(UnitTypeId.OVERLORD, 2), skip_until=RequiredSupply(13)),
                AutoOverLord(),
            ],
            CounterTerranTie([
                Step(RequiredAll([RequiredGas(100), RequiredUnitExists(UnitTypeId.ZERGLING, 6)]),
                     ActTech(UpgradeId.ZERGLINGMOVEMENTSPEED, UnitTypeId.SPAWNINGPOOL)),
                Step(RequiredGas(300), MorphLair()),

                Step(RequiredAll([RequiredGas(180), RequiredUnitExists(UnitTypeId.BANELING, 4),
                                  RequiredUnitReady(UnitTypeId.LAIR, 1)]),
                     ActTech(UpgradeId.CENTRIFICALHOOKS, UnitTypeId.BANELINGNEST)),

                Step(RequiredAll([RequiredGas(150), RequiredUnitExists(UnitTypeId.ROACH, 5),
                                  RequiredUnitReady(UnitTypeId.LAIR, 1)]),
                     ActTech(UpgradeId.GLIALRECONSTITUTION, UnitTypeId.ROACHWARREN)),
                bases,
                queens,
                drones,
                lings,
                roaches,
                hydras,
                banelings,
                PlanDistributeWorkers(1)
            ]),
            tactics
        ]
