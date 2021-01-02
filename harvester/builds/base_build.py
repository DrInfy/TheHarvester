from typing import List, Union, Tuple, Callable

from harvester.zerg_action import ZergAction
from sc2 import Result, Race
from sharpy.interfaces import IZoneManager, IGameAnalyzer
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
from sharpy.managers.extensions import BuildDetector
from sharpy.managers.core import EnemyUnitsManager
from tactics.ml.units import *


def numerize_race(race: Race):
    if race == Race.Protoss:
        return 0
    if race == Race.Zerg:
        return 1
    if race == Race.Terran:
        return 2
    return 3


class AllBuild(MlBuild):
    eu: EnemyUnitsManager
    bp: BuildDetector
    zm: IZoneManager
    game_analyzer: IGameAnalyzer

    def __init__(self):
        self.statesf: List[Callable[[], float]] = []
        self.statesr: List[float] = []
        self.attack = PlanZoneAttack(15)
        self.attack.attack_on_advantage = False
        orders = self.create_builds()
        self.init_funcs()
        # Dropping Hive tech for now
        super().__init__(len(self.statesf), 18, orders, result_multiplier=1)
        self.minimum_action_time = 5
        self.update_on_mineral_loss = False

    def init_funcs(self):
        self.statesf.append(lambda: self.ai.time)
        self.statesf.append(lambda: self.ai.minerals)
        self.statesf.append(lambda: self.ai.vespene)
        self.statesf.append(lambda: self.ai.supply_workers)
        self.statesf.append(lambda: self.ai.supply_army)
        self.statesf.append(lambda: self.zm.expansion_zones[0].paths[len(self.zm.expansion_zones) - 1].distance)

        self.add_score_funcs()
        self.statesf.append(lambda: len(self.ai.townhalls.filter(lambda u: not u.is_active)))

        self.statesf.append(lambda: numerize_race(self.ai.enemy_race))
        self.statesf.append(lambda: self.game_analyzer.our_power.air_power)
        self.statesf.append(lambda: self.game_analyzer.our_power.ground_power)
        self.statesf.append(lambda: self.game_analyzer.our_power.air_presence)
        self.statesf.append(lambda: self.game_analyzer.our_power.ground_presence)

        self.statesf.append(lambda: self.game_analyzer.enemy_power.air_power)
        self.statesf.append(lambda: self.game_analyzer.enemy_power.ground_power)
        self.statesf.append(lambda: self.game_analyzer.enemy_power.air_presence)
        self.statesf.append(lambda: self.game_analyzer.enemy_power.ground_presence)

        for upgrade in zerg_upgrades:
            self.statesf.append(lambda tmp=upgrade: self.get_ml_upgrade_progress([tmp]))

        self.statesf.append(
            lambda: self.get_ml_upgrade_progress(
                [
                    UpgradeId.ZERGMISSILEWEAPONSLEVEL1,
                    UpgradeId.ZERGMISSILEWEAPONSLEVEL2,
                    UpgradeId.ZERGMISSILEWEAPONSLEVEL3,
                ]
            )
        )
        self.statesf.append(
            lambda: self.get_ml_upgrade_progress(
                [UpgradeId.ZERGGROUNDARMORSLEVEL1, UpgradeId.ZERGGROUNDARMORSLEVEL2, UpgradeId.ZERGGROUNDARMORSLEVEL3]
            )
        )
        self.statesf.append(
            lambda: self.get_ml_upgrade_progress(
                [UpgradeId.ZERGMELEEWEAPONSLEVEL1, UpgradeId.ZERGMELEEWEAPONSLEVEL2, UpgradeId.ZERGMELEEWEAPONSLEVEL3]
            )
        )

        self.add_flier_upgrades()

        for unit_type in zerg_units:
            self.statesf.append(lambda tmp=unit_type: self.get_ordered_count(tmp))
            self.statesf.append(
                lambda tmp=unit_type: self.get_count(tmp, include_pending=False, include_not_ready=False)
            )

        for building_type in zerg_buildings:
            self.statesf.append(lambda tmp=building_type: self.get_ml_number(tmp))

        self.add_enemy_zerg()
        self.add_enemy_terran()
        self.add_enemy_protoss()

        for index in range(0, 5):
            self.zone_values(index)

        for index in range(-5, 0):
            self.zone_values(index)

    def zone_values(self, zone_index: int):
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].our_power.air_presence)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].our_power.ground_presence)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].our_power.air_power)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].our_power.ground_power)

        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].known_enemy_power.air_presence)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].known_enemy_power.ground_presence)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].known_enemy_power.air_power)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].known_enemy_power.ground_power)

        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].assaulting_enemy_power.air_presence)
        self.statesf.append(
            lambda: self.zone_manager.expansion_zones[zone_index].assaulting_enemy_power.ground_presence
        )
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].assaulting_enemy_power.air_power)
        self.statesf.append(lambda: self.zone_manager.expansion_zones[zone_index].assaulting_enemy_power.ground_power)

    def add_flier_upgrades(self):
        self.statesf.append(
            lambda: self.get_ml_upgrade_progress(
                [UpgradeId.ZERGFLYERWEAPONSLEVEL1, UpgradeId.ZERGFLYERWEAPONSLEVEL2, UpgradeId.ZERGFLYERWEAPONSLEVEL3]
            )
        )

        self.statesf.append(
            lambda: self.get_ml_upgrade_progress(
                [UpgradeId.ZERGFLYERARMORSLEVEL1, UpgradeId.ZERGFLYERARMORSLEVEL2, UpgradeId.ZERGFLYERARMORSLEVEL3]
            )
        )

    def add_score_funcs(self):
        self.statesf.append(lambda: self.ai.state.score.collection_rate_minerals)
        self.statesf.append(lambda: self.ai.state.score.collection_rate_vespene)

        self.statesf.append(lambda: self.ai.state.score.killed_minerals_army)
        self.statesf.append(lambda: self.ai.state.score.killed_minerals_economy)
        self.statesf.append(lambda: self.ai.state.score.killed_minerals_technology)
        self.statesf.append(lambda: self.ai.state.score.killed_minerals_upgrade)

        self.statesf.append(lambda: self.ai.state.score.killed_vespene_army)
        self.statesf.append(lambda: self.ai.state.score.killed_vespene_economy)
        self.statesf.append(lambda: self.ai.state.score.killed_vespene_technology)
        self.statesf.append(lambda: self.ai.state.score.killed_vespene_upgrade)

        self.statesf.append(lambda: self.ai.state.score.lost_minerals_army)
        self.statesf.append(lambda: self.ai.state.score.lost_minerals_economy)
        self.statesf.append(lambda: self.ai.state.score.lost_minerals_technology)
        self.statesf.append(lambda: self.ai.state.score.lost_minerals_upgrade)

        self.statesf.append(lambda: self.ai.state.score.lost_vespene_army)
        self.statesf.append(lambda: self.ai.state.score.lost_vespene_economy)
        self.statesf.append(lambda: self.ai.state.score.lost_vespene_technology)
        self.statesf.append(lambda: self.ai.state.score.lost_vespene_upgrade)

        self.statesf.append(lambda: self.ai.state.score.used_minerals_army)
        self.statesf.append(lambda: self.ai.state.score.used_minerals_economy)
        self.statesf.append(lambda: self.ai.state.score.used_minerals_technology)
        self.statesf.append(lambda: self.ai.state.score.used_minerals_upgrade)

        self.statesf.append(lambda: self.ai.state.score.used_vespene_army)
        self.statesf.append(lambda: self.ai.state.score.used_vespene_economy)
        self.statesf.append(lambda: self.ai.state.score.used_vespene_technology)
        self.statesf.append(lambda: self.ai.state.score.used_vespene_upgrade)

    def add_enemy_zerg(self):
        for unit_type in zerg_units:
            self.statesf.append(lambda: self.eu.unit_count(unit_type))

        for building_type in zerg_buildings:
            self.statesf.append(lambda: self.get_building_timing(building_type, 0))

        self.add_timings(UnitTypeId.HATCHERY, 6)

        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.EVOLUTIONCHAMBER, 1))
        self.statesf.append(lambda: self.get_building_timing(UnitTypeId.SPIRE, 1))
        self.add_timings(UnitTypeId.SPORECRAWLER, 5)
        self.add_timings(UnitTypeId.SPINECRAWLER, 5)

    def add_enemy_terran(self):
        for unit_type in terran_units:
            self.statesf.append(lambda: self.eu.unit_count(unit_type))

        for building_type in terran_buildings:
            self.statesf.append(lambda: self.get_building_timing(building_type, 0))

        self.add_timings(UnitTypeId.COMMANDCENTER, 6)
        self.add_timings(UnitTypeId.BARRACKS, 6)
        self.add_timings(UnitTypeId.FACTORY, 4)
        self.add_timings(UnitTypeId.STARPORT, 3)
        self.add_timings(UnitTypeId.SUPPLYDEPOT, 5)
        self.add_timings(UnitTypeId.BUNKER, 5)
        self.add_timings(UnitTypeId.MISSILETURRET, 5)

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
        self.add_timings(UnitTypeId.SHIELDBATTERY, 5)

    async def start(self, knowledge: "Knowledge"):
        await super().start(knowledge)
        self.eu: EnemyUnitsManager = self.knowledge.get_required_manager(EnemyUnitsManager)
        self.bp: BuildDetector = self.knowledge.get_required_manager(BuildDetector)
        self.zm: IZoneManager = self.knowledge.get_required_manager(IZoneManager)
        self.game_analyzer = self.knowledge.get_required_manager(IGameAnalyzer)
        # self.zone_manager.expansion_zones[0].paths[-1].distance

    def add_timings(self, unit_type: UnitTypeId, count: int):
        for index in range(1, count):
            self.statesf.append(lambda: self.get_building_timing(unit_type, index))

    def get_building_timing(self, unit_type: UnitTypeId, index: int) -> float:
        timing_list = self.bp.timings.get(unit_type, None)
        if timing_list is None or len(timing_list) <= index:
            return 0
        return timing_list[index]

    @property
    def score(self) -> float:
        score = 0
        # used = current
        # total_used_minerals_army = all value ever used

        score += self.ai.state.score.killed_minerals_army
        score += self.ai.state.score.killed_vespene_army
        score += self.ai.state.score.killed_vespene_economy
        score += self.ai.state.score.killed_minerals_economy * 2
        score += self.ai.state.score.killed_vespene_technology
        score += self.ai.state.score.killed_minerals_technology

        # score -= self.ai.state.score.lost_minerals_army
        # score -= self.ai.state.score.lost_vespene_army
        # score -= self.ai.state.score.lost_vespene_economy
        # score -= self.ai.state.score.lost_minerals_economy
        # score -= self.ai.state.score.lost_vespene_technology
        # score -= self.ai.state.score.lost_minerals_technology

        score = max(0, min(10000, score))  # Let's cap the score in order to not keep grinding enemy units forever

        # score += self.ai.state.score.used_minerals_army
        # score += self.ai.state.score.used_vespene_army
        for townhall in self.cache.own_townhalls:
            score += min(townhall.ideal_harvesters, townhall.assigned_harvesters) * 50

        for gas_building in self.ai.gas_buildings:
            score += min(gas_building.ideal_harvesters, gas_building.assigned_harvesters) * 50

        score += self.calc_upgrade_score()

        # score -= self.ai.state.score.lost_minerals_economy
        # score -= self.ai.state.score.lost_vespene_economy
        # score -= self.ai.state.score.lost_minerals_army
        # score -= self.ai.state.score.lost_vespene_army

        #
        # score += self.ai.state.score.used_minerals_technology
        # score += self.ai.state.score.used_vespene_technology
        #
        # score += self.ai.state.score.collection_rate_minerals
        # score += self.ai.state.score.collection_rate_vespene
        #
        self.reward = score / 100.0

        return super().score

    def calc_upgrade_score(self) -> float:
        value = 0

        def upgrade_value(score: float) -> float:
            return min(200, max(0, score - 150))

        for upgrade in [
            UpgradeId.ZERGMELEEWEAPONSLEVEL1,
            UpgradeId.ZERGMELEEWEAPONSLEVEL2,
            UpgradeId.ZERGMELEEWEAPONSLEVEL3,
        ]:
            value += upgrade_value(10 * self.game_analyzer.our_power.melee_power)
        for upgrade in [
            UpgradeId.ZERGMISSILEWEAPONSLEVEL1,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL2,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL3,
        ]:
            value += upgrade_value(
                10 * (self.game_analyzer.our_power.ground_power - self.game_analyzer.our_power.melee_power)
            )
        for upgrade in [
            UpgradeId.ZERGGROUNDARMORSLEVEL1,
            UpgradeId.ZERGGROUNDARMORSLEVEL2,
            UpgradeId.ZERGGROUNDARMORSLEVEL3,
        ]:
            value += upgrade_value(8 * self.game_analyzer.our_power.ground_presence)
        for upgrade in [
            UpgradeId.ZERGFLYERWEAPONSLEVEL1,
            UpgradeId.ZERGFLYERWEAPONSLEVEL2,
            UpgradeId.ZERGFLYERWEAPONSLEVEL3,
            UpgradeId.ZERGFLYERARMORSLEVEL1,
            UpgradeId.ZERGFLYERARMORSLEVEL2,
            UpgradeId.ZERGFLYERARMORSLEVEL3,
        ]:
            value += upgrade_value(10 * self.game_analyzer.our_power.air_presence)

        return value

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        if action == 0:
            return ("bases", (0, 255, 0))
        if action == 1:
            return ("drones", (0, 255, 0))
        if action == 2:
            return ("queens", (0, 255, 0))
        if action == 3:
            return ("lings", (255, 0, 0))
        if action == 4:
            return ("roaches", (255, 0, 0))
        if action == 5:
            return ("hydras", (255, 0, 0))
        if action == 6:
            return ("banelings", (255, 0, 0))
        if action == 7:
            return ("ravagers", (255, 0, 0))
        if action == 8:
            return ("lurkers", (255, 0, 255))
        if action == 9:
            return ("infestors", (255, 0, 255))
        if action == 10:
            return ("swarm hosts", (255, 0, 255))
        if action == 11:
            return ("mutalisks", (255, 0, 255))
        if action == 12:
            return ("corruptors", (255, 0, 255))
        if action == 13:
            return ("melee upgrades", (255, 255, 0))
        if action == 14:
            return ("ranged upgrades", (255, 255, 0))
        if action == 15:
            return ("armor upgrades", (255, 255, 0))
        if action == 16:
            return ("flier upgrades", (255, 255, 0))
        if action == 17:
            return ("spines", (255, 0, 255))
        if action == 18:
            return ("brood lords", (255, 0, 255))
        if action == 19:
            return ("vipers", (255, 0, 255))
        if action == 20:
            return ("ultralisks", (255, 0, 255))
        # if action == 20:
        #     return ("overseers", (255, 255, 0))
        return ("UNKNOWN " + str(action), (255, 255, 255))

    @property
    def state(self) -> List[Union[int, float]]:
        self.statesr.clear()
        for func in self.statesf:
            self.statesr.append(func())
        return self.statesr

    def on_end(self, game_result: Result):
        self.game_ended = True
        self.reward = self.score

        if game_result == Result.Victory:
            self.reward += 1000
            # self.reward += (
            #     min(5000, self.ai.state.score.collection_rate_minerals + self.ai.state.score.collection_rate_vespene)
            #     * 0.1
            # )  # self.result_multiplier
        else:
            ...  # do nothing
            # score = -100
            # score += min(5000, self.ai.state.score.used_minerals_economy)
            # score += min(10000, self.ai.state.score.killed_minerals_army + self.ai.state.score.killed_vespene_army)
            # score += min(15 * 60, self.ai.time) * 5  # max 4500
            # self.reward += score * 0.01

            # score += self.ai.state.score.used_minerals_army
            # score += self.ai.state.score.lost_minerals_army
            # score += self.ai.state.score.used_vespene_army
            # score += self.ai.state.score.lost_vespene_army
            # score += self.ai.state.score.killed_vespene_economy
            # score += self.ai.state.score.killed_minerals_economy
            # self.reward = 0
        final_reward = self.reward * self.result_multiplier
        self.print(f"FinalScore: {final_reward}")
        self.agent.on_end(self.state, final_reward)

    def create_builds(self) -> List[Union[ActBase, List[ActBase]]]:
        tactics = SequentialList(
            [
                OverlordScout(ScoutLocation.scout_enemy2(), ScoutLocation.scout_enemy3()),
                OverlordScout(ScoutLocation.scout_own3(), ScoutLocation.scout_own4()),
                OverlordScout(ScoutLocation.scout_enemy3(), ScoutLocation.scout_enemy4()),
                OverlordScout(ScoutLocation.scout_center_ol_spot()),
                OverlordScout(ScoutLocation.scout_own_natural_ol_spot()),
                OverlordScout(ScoutLocation.scout_own5(), ScoutLocation.scout_own3(), ScoutLocation.scout_own4(),),
                OverlordScout(ScoutLocation.scout_own5(), ScoutLocation.scout_own2()),
                Step(
                    None,
                    Scout(
                        UnitTypeId.ZERGLING,
                        2,
                        ScoutLocation.scout_main(),
                        ScoutLocation.scout_enemy1(),
                        ScoutLocation.scout_enemy2(),
                    ),
                    skip_until=Time(4 * 60),
                ),
                Step(
                    None,
                    Scout(
                        UnitTypeId.ZERGLING,
                        2,
                        ScoutLocation.scout_main(),
                        ScoutLocation.scout_enemy1(),
                        ScoutLocation.scout_enemy2(),
                    ),
                    skip_until=Time(8 * 60),
                ),
                Step(None, WorkerScout(), skip_until=Time(40)),
                SpreadCreep(),
                InjectLarva(),
                PlanHeatOverseer(),
                PlanWorkerOnlyDefense(),
                PlanZoneDefense(),
                PlanZoneGather(),
                self.attack,
                PlanFinishEnemy(),
            ]
        )
        gas = SequentialList(
            [
                Step(None, BuildGas(2), skip=Gas(200), skip_until=Supply(25, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(3), skip=Gas(200), skip_until=Supply(40, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(4), skip=Gas(200), skip_until=Supply(50, supply_type=SupplyType.Workers),),
                Step(
                    Minerals(1000), BuildGas(6), skip=Gas(200), skip_until=Supply(50, supply_type=SupplyType.Workers),
                ),
                Step(
                    Minerals(2000), BuildGas(8), skip=Gas(200), skip_until=Supply(50, supply_type=SupplyType.Workers),
                ),
            ]
        )

        heavy_gas = SequentialList(
            [
                Step(None, BuildGas(2), skip=Gas(300), skip_until=Supply(20, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(3), skip=Gas(300), skip_until=Supply(30, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(4), skip=Gas(300), skip_until=Supply(40, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(5), skip=Gas(300), skip_until=Supply(50, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(6), skip=Gas(300), skip_until=Supply(60, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(7), skip=Gas(300), skip_until=Supply(65, supply_type=SupplyType.Workers),),
                Step(None, BuildGas(8), skip=Gas(300), skip_until=Supply(70, supply_type=SupplyType.Workers),),
            ]
        )

        auto_anti_cloak = [
            Step(UnitReady(UnitTypeId.SPAWNINGPOOL), None),
            Step(RequireCustom(lambda k: k.enemy_units_manager.enemy_cloak_trigger), None),
            # DefensiveBuilding(UnitTypeId.SPORECRAWLER, DefensePosition.CenterMineralLine),
            MorphLair(),
            Step(UnitReady(UnitTypeId.LAIR), MorphOverseer(2)),
        ]

        pool = GridBuilding(UnitTypeId.SPAWNINGPOOL)
        bases = Step(lambda k: self.action == 0, Expand(10))
        drones = Step(lambda k: self.action == 1, ZergUnit(UnitTypeId.DRONE, 90))

        queens = Step(lambda k: self.action == 2, SequentialList([pool, BuildOrder([ZergUnit(UnitTypeId.QUEEN),])]),)

        lings = Step(
            lambda k: self.action == ZergAction.Lings,
            SequentialList(
                [
                    pool,
                    BuildOrder(
                        [
                            BuildGas(1),
                            Step(Gas(90), Tech(UpgradeId.ZERGLINGMOVEMENTSPEED, UnitTypeId.SPAWNINGPOOL,),),
                            ZergUnit(UnitTypeId.ZERGLING),
                        ]
                    ),
                ]
            ),
        )

        banelings = Step(
            lambda k: self.action == ZergAction.Banelings,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    Step(UnitReady(UnitTypeId.SPAWNINGPOOL), GridBuilding(UnitTypeId.BANELINGNEST),),
                    gas,
                    ZergUnit(UnitTypeId.BANELING),
                ]
            ),
        )

        roaches = Step(
            lambda k: self.action == ZergAction.Roaches,
            SequentialList(
                [
                    pool,
                    Step(UnitReady(UnitTypeId.SPAWNINGPOOL), GridBuilding(UnitTypeId.ROACHWARREN),),
                    BuildGas(1),
                    gas,
                    ZergUnit(UnitTypeId.ROACH),
                ]
            ),
        )

        ravagers = Step(
            lambda k: self.action == ZergAction.Ravagers,
            SequentialList(
                [
                    pool,
                    Step(
                        UnitReady(UnitTypeId.SPAWNINGPOOL),
                        PositionBuilding(UnitTypeId.ROACHWARREN, DefensePosition.BehindMineralLineRight, 0,),
                    ),
                    BuildGas(1),
                    heavy_gas,
                    ZergUnit(UnitTypeId.RAVAGER),
                ]
            ),
        )

        hydras = Step(
            lambda k: self.action == ZergAction.Hydras,
            SequentialList(
                pool,
                BuildGas(1),
                BuildOrder(
                    MorphLair(), GridBuilding(UnitTypeId.HYDRALISKDEN, 1), heavy_gas, ZergUnit(UnitTypeId.HYDRALISK),
                ),
            ),
        )

        lurkers = Step(
            lambda k: self.action == ZergAction.Lurkers,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    gas,
                    MorphLair(),
                    GridBuilding(UnitTypeId.HYDRALISKDEN, 1),
                    BuildOrder([GridBuilding(UnitTypeId.LURKERDENMP, 1), ZergUnit(UnitTypeId.LURKERMP), heavy_gas,]),
                ]
            ),
        )

        infestors = Step(
            lambda k: self.action == ZergAction.Infestors,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    gas,
                    MorphLair(),
                    GridBuilding(UnitTypeId.INFESTATIONPIT, 1),
                    BuildOrder([ZergUnit(UnitTypeId.INFESTOR), heavy_gas]),
                ]
            ),
        )

        swarmhosts = Step(
            lambda k: self.action == ZergAction.SwarmHosts,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    gas,
                    MorphLair(),
                    GridBuilding(UnitTypeId.INFESTATIONPIT, 1),
                    BuildOrder([ZergUnit(UnitTypeId.SWARMHOSTMP), heavy_gas]),
                ]
            ),
        )

        mutalisks = Step(
            lambda k: self.action == ZergAction.Mutalisks,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    gas,
                    MorphLair(),
                    GridBuilding(UnitTypeId.SPIRE, 1),
                    BuildOrder([ZergUnit(UnitTypeId.MUTALISK), heavy_gas]),
                ]
            ),
        )

        corruptors = Step(
            lambda k: self.action == ZergAction.Corruptors,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    gas,
                    MorphLair(),
                    GridBuilding(UnitTypeId.SPIRE, 1),
                    BuildOrder([ZergUnit(UnitTypeId.CORRUPTOR), heavy_gas]),
                ]
            ),
        )

        melee_upgrades = Step(
            lambda k: self.action == ZergAction.MeleeUpgrades,
            BuildOrder(
                [
                    Step(
                        All([Gas(180), UnitExists(UnitTypeId.BANELING, 4), UnitReady(UnitTypeId.LAIR, 1),]),
                        Tech(UpgradeId.CENTRIFICALHOOKS, UnitTypeId.BANELINGNEST),
                    ),
                    SequentialList(
                        [
                            GridBuilding(UnitTypeId.EVOLUTIONCHAMBER, 1),
                            Tech(UpgradeId.ZERGMELEEWEAPONSLEVEL1, UnitTypeId.EVOLUTIONCHAMBER,),
                            MorphLair(),
                            Step(
                                UnitReady(UnitTypeId.LAIR),
                                Tech(UpgradeId.ZERGMELEEWEAPONSLEVEL2, UnitTypeId.EVOLUTIONCHAMBER,),
                            ),
                            MorphHive(),
                            Step(
                                UnitReady(UnitTypeId.HIVE),
                                Tech(UpgradeId.ZERGMELEEWEAPONSLEVEL3, UnitTypeId.EVOLUTIONCHAMBER,),
                            ),
                        ]
                    ),
                    Step(
                        lambda k: k.unit_cache.own(UnitTypeId.EVOLUTIONCHAMBER).ready.amount > 0
                        and k.unit_cache.own(UnitTypeId.EVOLUTIONCHAMBER).idle.amount == 0,
                        GridBuilding(UnitTypeId.EVOLUTIONCHAMBER, 2),
                    ),
                    Step(UnitReady(UnitTypeId.HIVE), Tech(UpgradeId.ZERGLINGATTACKSPEED, UnitTypeId.SPAWNINGPOOL),),
                    Step(
                        UnitReady(UnitTypeId.ULTRALISKCAVERN),
                        Tech(UpgradeId.CHITINOUSPLATING, UnitTypeId.ULTRALISKCAVERN),
                    ),
                    Step(
                        UnitReady(UnitTypeId.ULTRALISKCAVERN),
                        Tech(UpgradeId.ANABOLICSYNTHESIS, UnitTypeId.ULTRALISKCAVERN),
                    ),
                    BuildGas(1),
                    gas,
                    Step(
                        All(
                            Minerals(300),
                            UnitReady(UnitTypeId.LAIR),
                            Supply(100),
                            TechReady(UpgradeId.ZERGMELEEWEAPONSLEVEL2),
                        ),
                        SequentialList(GridBuilding(UnitTypeId.INFESTATIONPIT), MorphHive(),),
                    ),
                ]
            ),
        )

        ranged_upgrades = Step(
            lambda k: self.action == ZergAction.RangedUpgrades,
            BuildOrder(
                [
                    Step(
                        UnitReady(UnitTypeId.ROACHWARREN), Tech(UpgradeId.GLIALRECONSTITUTION, UnitTypeId.ROACHWARREN),
                    ),
                    SequentialList(
                        Tech(UpgradeId.ZERGMISSILEWEAPONSLEVEL1, UnitTypeId.EVOLUTIONCHAMBER),
                        MorphLair(),
                        Step(
                            UnitReady(UnitTypeId.LAIR),
                            Tech(UpgradeId.ZERGMISSILEWEAPONSLEVEL2, UnitTypeId.EVOLUTIONCHAMBER,),
                        ),
                        MorphHive(),
                        Step(
                            UnitReady(UnitTypeId.HIVE),
                            Tech(UpgradeId.ZERGMISSILEWEAPONSLEVEL3, UnitTypeId.EVOLUTIONCHAMBER,),
                        ),
                    ),
                    Step(
                        All([UnitReady(UnitTypeId.HYDRALISKDEN), UnitExists(UnitTypeId.HYDRALISK, 3),]),
                        SequentialList(
                            Tech(UpgradeId.EVOLVEGROOVEDSPINES),
                            Tech(UpgradeId.EVOLVEMUSCULARAUGMENTS),  # HYDRALISKSPEED
                        ),
                    ),
                    GridBuilding(UnitTypeId.EVOLUTIONCHAMBER, 1),
                    Step(
                        lambda k: k.unit_cache.own(UnitTypeId.EVOLUTIONCHAMBER).ready.amount > 0
                        and k.unit_cache.own(UnitTypeId.EVOLUTIONCHAMBER).idle.amount == 0,
                        GridBuilding(UnitTypeId.EVOLUTIONCHAMBER, 2),
                    ),
                    BuildGas(1),
                    gas,
                    Step(
                        All(
                            Minerals(300),
                            UnitReady(UnitTypeId.LAIR),
                            Supply(100),
                            TechReady(UpgradeId.ZERGMISSILEWEAPONSLEVEL2),
                        ),
                        SequentialList(GridBuilding(UnitTypeId.INFESTATIONPIT), MorphHive(),),
                    ),
                ]
            ),
        )

        armor_upgrades = Step(
            lambda k: self.action == ZergAction.ArmorUpgrades,
            BuildOrder(
                GridBuilding(UnitTypeId.EVOLUTIONCHAMBER, 1),
                SequentialList(
                    Tech(UpgradeId.ZERGGROUNDARMORSLEVEL1, UnitTypeId.EVOLUTIONCHAMBER),
                    MorphLair(),
                    Step(
                        UnitReady(UnitTypeId.LAIR),
                        Tech(UpgradeId.ZERGGROUNDARMORSLEVEL2, UnitTypeId.EVOLUTIONCHAMBER,),
                    ),
                    MorphHive(),
                    Step(
                        UnitReady(UnitTypeId.HIVE),
                        Tech(UpgradeId.ZERGGROUNDARMORSLEVEL3, UnitTypeId.EVOLUTIONCHAMBER,),
                    ),
                ),
                Step(
                    lambda k: k.unit_cache.own(UnitTypeId.EVOLUTIONCHAMBER).ready.amount > 0
                    and k.unit_cache.own(UnitTypeId.EVOLUTIONCHAMBER).idle.amount == 0,
                    GridBuilding(UnitTypeId.EVOLUTIONCHAMBER, 2),
                ),
                BuildGas(1),
                gas,
                Step(
                    All(
                        Minerals(300),
                        UnitReady(UnitTypeId.LAIR),
                        Supply(100),
                        TechReady(UpgradeId.ZERGGROUNDARMORSLEVEL2),
                    ),
                    SequentialList(GridBuilding(UnitTypeId.INFESTATIONPIT), MorphHive(),),
                ),
            ),
        )

        air_upgrades = Step(
            lambda k: self.action == ZergAction.AirUpgrades,
            SequentialList(
                pool,
                BuildGas(1),
                gas,
                MorphLair(),
                GridBuilding(UnitTypeId.SPIRE, 1),
                Tech(UpgradeId.ZERGFLYERWEAPONSLEVEL1),
                Tech(UpgradeId.ZERGFLYERWEAPONSLEVEL2),
                Step(None, Tech(UpgradeId.ZERGFLYERWEAPONSLEVEL3), skip_until=UnitReady(UnitTypeId.HIVE)),
                Tech(UpgradeId.ZERGFLYERARMORSLEVEL1),
                Tech(UpgradeId.ZERGFLYERARMORSLEVEL2),
                GridBuilding(UnitTypeId.INFESTATIONPIT),
                MorphHive(),
                Tech(UpgradeId.ZERGFLYERARMORSLEVEL3),
            ),
        )

        spines = Step(
            lambda k: self.action == ZergAction.Spines,
            SequentialList(
                [
                    Step(
                        None,
                        DefensiveBuilding(UnitTypeId.SPINECRAWLER, DefensePosition.FarEntrance, 0, 4),
                        skip=lambda k: k.townhalls.amount > 1,
                    ),
                    Step(
                        None,
                        DefensiveBuilding(UnitTypeId.SPINECRAWLER, DefensePosition.FarEntrance, 1, 1),
                        skip_until=lambda k: k.townhalls.amount >= 2,
                    ),
                    Step(
                        None,
                        DefensiveBuilding(UnitTypeId.SPINECRAWLER, DefensePosition.FarEntrance, 1, 3),
                        skip_until=lambda k: k.townhalls.amount == 2,
                    ),
                    Step(
                        lambda k: k.townhalls.amount > 2,
                        DefensiveBuilding(UnitTypeId.SPINECRAWLER, DefensePosition.Entrance, 2, 1),
                    ),
                    Step(
                        lambda k: k.townhalls.amount > 3,
                        DefensiveBuilding(UnitTypeId.SPINECRAWLER, DefensePosition.Entrance, 3, 1),
                    ),
                ]
            ),
        )

        broodlords = Step(
            lambda k: self.action == ZergAction.BroodLords,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    BuildOrder(
                        [
                            MorphLair(),
                            GridBuilding(UnitTypeId.SPIRE, 1),
                            GridBuilding(UnitTypeId.INFESTATIONPIT, 1),
                            MorphHive(),
                            MorphGreaterSpire(),
                            ZergUnit(UnitTypeId.BROODLORD),
                            heavy_gas,
                        ]
                    ),
                ]
            ),
        )

        vipers = Step(
            lambda k: self.action == ZergAction.Vipers,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    BuildOrder(
                        [
                            MorphLair(),
                            GridBuilding(UnitTypeId.INFESTATIONPIT, 1),
                            MorphHive(),
                            ZergUnit(UnitTypeId.VIPER),
                            heavy_gas,
                        ]
                    ),
                ]
            ),
        )

        ultralisks = Step(
            lambda k: self.action == ZergAction.UltraLisks,
            SequentialList(
                [
                    pool,
                    BuildGas(1),
                    BuildOrder(
                        [
                            MorphLair(),
                            GridBuilding(UnitTypeId.INFESTATIONPIT, 1),
                            MorphHive(),
                            GridBuilding(UnitTypeId.ULTRALISKCAVERN, 1),
                            ZergUnit(UnitTypeId.VIPER),
                            heavy_gas,
                        ]
                    ),
                ]
            ),
        )

        return [
            [ZergUnit(UnitTypeId.OVERLORD, 1), Step(Supply(13), ZergUnit(UnitTypeId.OVERLORD, 2)), AutoOverLord(),],
            CounterTerranTie(
                [
                    auto_anti_cloak,
                    bases,
                    queens,
                    drones,
                    lings,
                    roaches,
                    hydras,
                    banelings,
                    ravagers,
                    lurkers,
                    infestors,
                    swarmhosts,
                    mutalisks,
                    corruptors,
                    broodlords,
                    vipers,
                    ultralisks,
                    spines,
                    melee_upgrades,
                    armor_upgrades,
                    ranged_upgrades,
                    air_upgrades,
                    DistributeWorkers(1, aggressive_gas_fill=True),
                ]
            ),
            tactics,
        ]
