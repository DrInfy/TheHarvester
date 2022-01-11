from typing import List, Union, Tuple

from sc2 import UnitTypeId
from sharpy.general.path import Path
from sharpy.managers.core.roles import UnitTask
from sharpy.plans import SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from tactics.ml.ml_build import MlBuild

num_distraction_workers: int = 1
num_useless_workers: int = 5  # to avoid winning by chance we make some workers useless


class WorkerDistraction_v0(MlBuild):
    """Intended to be played against the WorkerRushDefender dummy bot.
    This build negatively handicaps the learning bot's economy and then uses agent decisions to run a scout worker toward the enemy.
    WorkerRushDefender is coded to be distracted from mining when the scout worker is near it's base.
    The learning bot should only be able to win via this distraction method."""

    STATE_SIZE = 3
    ACTION_SIZE = 2

    def __init__(self):
        super().__init__(WorkerDistraction_v0.STATE_SIZE,
                         WorkerDistraction_v0.ACTION_SIZE,
                         self.create_plan(),
                         result_multiplier=1.0)
        self.distraction_worker_tags: List[int] = []
        self.useless_worker_tags: List[int] = []
        self.is_dead = False

    @property
    def state(self) -> List[Union[int, float]]:
        path_distance_between_bases = Path(
            self.knowledge.pathing_manager.path_finder_terrain.find_path(
                self.ai.enemy_start_locations[0], self.ai.start_location)).distance

        # state flags
        self.is_dead = True
        enemy_workers_distracted: int = 0
        distance_ratio_to_enemy_main: float = 0.0

        # workers alive
        workers = self.ai.workers.tags_in(self.distraction_worker_tags).sorted_by_distance_to(
            self.ai.enemy_start_locations[0])
        if len(workers) > 0:
            self.is_dead = False

            if len(self.ai.enemy_units.of_type(UnitTypeId.DRONE).closer_than(5, workers[0].position)) > 0:
                enemy_workers_distracted = 1

            path_distance_to_enemy_main = Path(self.knowledge.pathing_manager.path_finder_terrain.find_path(
                self.ai.enemy_start_locations[0], workers[0].position)).distance
            distance_ratio_to_enemy_main = path_distance_to_enemy_main / path_distance_between_bases

        return [self.is_dead, enemy_workers_distracted, distance_ratio_to_enemy_main]

    @property
    def score(self) -> float:
        self.reward = 0

        workers = self.ai.workers.tags_in(self.distraction_worker_tags).sorted_by_distance_to(
            self.ai.enemy_start_locations[0])

        # enemy workers not mining
        if len(workers) > 0:
            enemy_drones = self.ai.enemy_units.of_type(UnitTypeId.DRONE).closer_than(10, workers[0].position)
            # not_mining_count = len(enemy_drones)
            # self.reward += not_mining_count
            self.reward = 1 if len(enemy_drones) > 0 else 0

        # Encourage attacking when we have workers
        # if len(self.ai.workers.tags_in(self.distraction_worker_tags)) > 0:
        #     self.reward += self.action  # 1 == attacking, 0 == retreating.
        return self.reward

    async def execute(self) -> bool:
        if self.is_dead:
            return True  # give up

        return await super().execute()

    async def start(self, knowledge: 'Knowledge'):
        await super().start(knowledge)
        distraction_workers = self.ai.workers.closest_n_units(self.ai.enemy_start_locations[0], num_distraction_workers)
        for worker in distraction_workers:
            self.knowledge.roles.set_task(UnitTask.Scouting, worker)
            self.distraction_worker_tags.append(worker.tag)

        useless_workers = self.ai.workers.tags_not_in(self.distraction_worker_tags) \
            .closest_n_units(self.ai.enemy_start_locations[0], num_useless_workers)
        for worker in useless_workers:
            self.knowledge.roles.set_task(UnitTask.Reserved, worker)
            self.useless_worker_tags.append(worker.tag)

    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        if self.is_dead:
            return "ACT: DEAD", (255, 255, 255)
        if action == 1:
            return ("ACT: ATTACK", (255, 0, 0))
        if action == 0:
            return ("ACT: RETREAT", (0, 255, 0))

        return super().get_action_name_color(action)

    def attack(self) -> bool:
        for worker in self.ai.workers.tags_in(self.distraction_worker_tags):
            worker.attack(self.ai.enemy_start_locations[0])
        return True

    def retreat(self):
        for worker in self.ai.workers.tags_in(self.distraction_worker_tags):
            worker.move(self.ai.start_location)
        return True

    def make_workers_useless(self):
        """Try and avoid use winning by accident by making some workers useless"""
        for worker in self.ai.workers.tags_in(self.useless_worker_tags):
            worker.move(self.ai.start_location)
        return True

    def create_plan(self) -> List[Union[ActBase, List[ActBase]]]:
        return [
            SequentialList([
                ActUnit(UnitTypeId.DRONE, UnitTypeId.LARVA, 14),
                Expand(2),
                ActBuilding(UnitTypeId.SPAWNINGPOOL, 1),
                ActUnit(UnitTypeId.OVERLORD, UnitTypeId.LARVA, 2),
                ActUnit(UnitTypeId.QUEEN, UnitTypeId.HATCHERY, 1),
                ActUnit(UnitTypeId.ZERGLING, UnitTypeId.LARVA, 200),
            ]),
            SequentialList(
                [
                    ActCustom(self.make_workers_useless),
                    ActCustom(lambda: self.attack() if self.action == 1 else self.retreat()),
                    DistributeWorkers(),
                    PlanZoneDefense(),
                    AutoOverLord(),
                    InjectLarva(),
                    PlanZoneGather(),
                    PlanZoneAttack(10),
                    PlanFinishEnemy(),
                ])
        ]
