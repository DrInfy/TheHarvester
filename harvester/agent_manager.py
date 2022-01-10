from random import randint
from typing import Optional

from harvester.zerg_action import ZergAction
from sc2 import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.interfaces import IGameAnalyzer, IEnemyUnitsManager
from sharpy.managers.extensions.game_states import AirArmy
from sharpy.managers.extensions.game_states.advantage import at_least_disadvantage, at_least_advantage
from tactics.ml.base_agent_manager import BaseAgentManager
from tactics.ml.ml_build import MlBuild


class ZergAgentManager(BaseAgentManager):
    game_analyzer: IGameAnalyzer
    enemy_units_manager: IEnemyUnitsManager

    def __init__(self, agent: str, build_str: str, build: MlBuild, shared_global_vars: dict, **kwargs) -> None:
        super().__init__(agent, build_str, build, shared_global_vars, **kwargs)
        self.hatcheryworkers1 = randint(13, 22)
        self.poolworkers = randint(16, 22)
        self.hatcheryworkers2 = randint(30, 50)
        self.cap_workers = randint(45, 90)
        self.mid_game_army = randint(25, 90)
        self.mid_game_hydras = randint(0, 2) * 8
        self.hydra_time = (randint(0, 4) + 5) * 60
        self.max_queens = randint(1, 3) * 3
        self.corruptors = randint(0, 5) > 3
        self.upgrades_start = randint(5, 12) * 60
        self.banelings = randint(0, 5) > 3

    async def start(self, knowledge: "Knowledge"):
        await super().start(knowledge)
        self.game_analyzer = knowledge.get_required_manager(IGameAnalyzer)
        self.enemy_units_manager = knowledge.get_required_manager(IEnemyUnitsManager)

    def scripted_action(self) -> int:
        if self.ai.minerals > 1000 and self.ai.time % 5 > 3 and self.ai.supply_used < 199:
            if self.ai.supply_workers < 50 and self.ai.vespene < 50:
                return ZergAction.Drones
            return self.build_army()

        if self.ai.supply_workers < self.hatcheryworkers1 and self.ai.supply_workers < self.poolworkers:
            return ZergAction.Drones

        if self.ai.supply_workers >= self.hatcheryworkers1 and self.cache.own_townhalls().amount == 1:
            return ZergAction.Bases

        if (
            self.ai.already_pending(UnitTypeId.QUEEN) + self.cache.own(UnitTypeId.QUEEN).amount
            < min(self.max_queens, self.cache.own_townhalls().amount)
            and self.cache.own_townhalls.idle
        ):
            return ZergAction.Queens

        if (
            self.ai.townhalls.amount == 2
            and self.ai.time < 5 * 60
            and self.game_analyzer.army_at_least_small_disadvantage
        ):
            # Defense vs rush
            if self.cache.own(UnitTypeId.SPINECRAWLER).amount == 0:
                return ZergAction.Spines
            return ZergAction.Lings

        upgrade = self.upgrades()
        if upgrade:
            return upgrade

        if (
            self.game_analyzer.army_at_least_small_disadvantage
            and self.game_analyzer.our_income_advantage in at_least_advantage
        ):
            return self.build_army()

        if self.ai.supply_workers >= self.hatcheryworkers2 and self.cache.own_townhalls().amount == 2:
            return ZergAction.Bases

        if self.game_analyzer.our_income_advantage in at_least_disadvantage:
            return self.build_economy()

        if self.game_analyzer.our_army_predict in at_least_disadvantage:
            return self.build_army()

        if self.ai.time < 8 * 60 and self.ai.supply_army < self.mid_game_army:
            return self.build_army()

        if self.ai.supply_workers < self.cap_workers:
            return self.build_economy()

        return self.build_army()

    def build_economy(self) -> ZergAction:
        idle_space = 0
        for townhall in self.ai.townhalls:
            idle_space += townhall.ideal_harvesters - townhall.assigned_harvesters

        if idle_space > 10:
            if self.ai.supply_used >= 200:
                action = self.upgrades()
                if action:
                    return action
            return ZergAction.Drones

        return ZergAction.Bases

    def upgrades(self) -> Optional[ZergAction]:
        if self.ai.time < self.upgrades_start or self.ai.time % 10 > 4:
            return None

        def allow_melee_upgrade():
            return (
                self.ai.already_pending_upgrade(UpgradeId.ZERGMELEEWEAPONSLEVEL1) % 1 == 0
                and self.ai.already_pending_upgrade(UpgradeId.ZERGMELEEWEAPONSLEVEL2) % 1 == 0
                and self.ai.already_pending_upgrade(UpgradeId.ZERGMELEEWEAPONSLEVEL3) == 0
            )

        def allow_ranged_upgrade():
            return (
                self.ai.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL1) % 1 == 0
                and self.ai.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL2) % 1 == 0
                and self.ai.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL3) == 0
            )

        def allow_armor_upgrade():
            return (
                self.ai.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL1) % 1 == 0
                and self.ai.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL2) % 1 == 0
                and self.ai.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL3) == 0
            )

        if self.ai.time > self.upgrades_start:
            melee_count = self.cache.own(UnitTypeId.ZERGLING).amount + self.cache.own(UnitTypeId.BANELING).amount
            ranged_count = (
                self.cache.own(UnitTypeId.ROACH).amount
                + self.cache.own(UnitTypeId.HYDRALISK).amount
                + self.cache.own(UnitTypeId.RAVAGER).amount
            )
            if (
                self.cache.own(UnitTypeId.SPIRE).idle
                and self.ai.already_pending_upgrade(UpgradeId.ZERGFLYERARMORSLEVEL3) == 0
                and self.game_analyzer.our_power.air_presence > 5
            ):
                return ZergAction.AirUpgrades

            evos = self.cache.own(UnitTypeId.EVOLUTIONCHAMBER)
            if not evos:
                if melee_count * 2 > ranged_count:
                    if allow_melee_upgrade():
                        return ZergAction.MeleeUpgrades
                else:
                    if allow_ranged_upgrade():
                        return ZergAction.RangedUpgrades
            if evos.idle:
                if melee_count * 2 > ranged_count:
                    if allow_melee_upgrade():
                        return ZergAction.MeleeUpgrades
                else:
                    if allow_ranged_upgrade():
                        return ZergAction.RangedUpgrades
            if (evos.amount < 2 or evos.idle) and allow_armor_upgrade():
                return ZergAction.ArmorUpgrades
        return None

    def build_army(self) -> ZergAction:
        larva = self.cache.own(UnitTypeId.LARVA).amount

        if self.enemy_units_manager.unit_count(UnitTypeId.VOIDRAY) > 2:
            if (
                self.cache.own(UnitTypeId.HYDRALISK).amount
                < self.enemy_units_manager.unit_count(UnitTypeId.VOIDRAY) * 2
            ):
                return ZergAction.Hydras

        if self.enemy_units_manager.unit_count(UnitTypeId.BANSHEE) > 1:
            if (
                self.corruptors
                and self.cache.own(UnitTypeId.MUTALISK).amount
                < self.enemy_units_manager.unit_count(UnitTypeId.BANSHEE) * 2
            ):
                return ZergAction.Mutalisks

            if (
                not self.corruptors
                and self.cache.own(UnitTypeId.HYDRALISK).amount
                < self.enemy_units_manager.unit_count(UnitTypeId.BANSHEE) * 2
            ):
                return ZergAction.Hydras

        if self.game_analyzer.enemy_air == AirArmy.AllAir or self.game_analyzer.enemy_air == AirArmy.AlmostAllAir:
            if not self.corruptors:
                if len(self.cache.own(UnitTypeId.HYDRALISK)) * 0.1 > len(self.cache.own(UnitTypeId.INFESTOR)) + 2:
                    return ZergAction.Infestors
                return ZergAction.Hydras
            return ZergAction.Corruptors

        if self.game_analyzer.enemy_air == AirArmy.Mixed:
            if not self.corruptors:
                if len(self.cache.own(UnitTypeId.HYDRALISK)) > len(self.cache.own(UnitTypeId.ROACH)):
                    return ZergAction.Roaches
                return ZergAction.Hydras
            if len(self.cache.own(UnitTypeId.CORRUPTOR)) > len(self.cache.own(UnitTypeId.ROACH)):
                return ZergAction.Roaches
            return ZergAction.Corruptors

        if self.ai.time < 240 or (self.cache.own(UnitTypeId.ZERGLING).amount < 12 and larva < 2):
            return ZergAction.Lings

        if self.banelings and (
            self.enemy_units_manager.unit_count(UnitTypeId.ZERGLING) > 25
            or self.enemy_units_manager.unit_count(UnitTypeId.MARINE) > 15
            or self.enemy_units_manager.unit_count(UnitTypeId.ZEALOT) > 10
        ):
            if self.cache.own(UnitTypeId.BANELING).amount < 10:
                return ZergAction.Banelings

        if (
            self.enemy_units_manager.unit_count(UnitTypeId.MARAUDER) > 6
            or self.enemy_units_manager.unit_count(UnitTypeId.STALKER) > 5
        ):
            if self.cache.own(UnitTypeId.ZERGLING).amount < 30:
                return ZergAction.Lings

        if (
            self.enemy_units_manager.unit_count(UnitTypeId.BATTLECRUISER) > 1
            or self.enemy_units_manager.unit_count(UnitTypeId.TEMPEST) > 1
            or self.enemy_units_manager.unit_count(UnitTypeId.CARRIER) > 1
            or self.enemy_units_manager.unit_count(UnitTypeId.BROODLORD) > 1
        ):
            if self.cache.own(UnitTypeId.CORRUPTOR).amount < 5:
                return ZergAction.Corruptors

        if self.ai.time > self.hydra_time and self.mid_game_hydras < len(self.cache.own(UnitTypeId.HYDRALISK)):
            return ZergAction.Hydras

        if (
            self.ai.vespene > 100
            and self.game_analyzer.enemy_air < AirArmy.SomeAir
            and len(self.cache.own(UnitTypeId.HYDRALISK)) > 6
            and len(self.cache.own(UnitTypeId.LURKER)) < 5
        ):
            return ZergAction.Lurkers

        if (
            self.ai.vespene > 100
            and len(self.cache.own(UnitTypeId.ROACH)) > len(self.cache.own(UnitTypeId.RAVAGER)) + 2
        ):
            return ZergAction.Ravagers

        return ZergAction.Roaches
