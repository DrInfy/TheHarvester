from typing import List, Union, Tuple

from sc2 import UnitTypeId
from sharpy.managers.core.roles import UnitTask
from sharpy.plans import SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from tactics.ml.ml_build import MlBuild

num_distraction_workers: int = 6

class WorkerDistraction_v0(MlBuild):
    STATE_SIZE = 1
    ACTION_SIZE = 2

    def __init__(self):
        super().__init__(WorkerDistraction_v0.STATE_SIZE,
                         WorkerDistraction_v0.ACTION_SIZE,
                         self.create_plan(),
                         result_multiplier=1.0)
        self.distraction_worker_tags: List[int] = []
        self.is_dead = False

    @property
    def state(self) -> List[Union[int, float]]:
        # workers alive
        workers = self.ai.workers.tags_in(self.distraction_worker_tags).sorted_by_distance_to(
            self.ai.enemy_start_locations[0])
        if len(workers) > 0:
            return [1] if len(self.ai.enemy_units.of_type(UnitTypeId.DRONE).closer_than(5, workers[0].position)) > 0 \
                       else [0]
        else:
            self.is_dead = True
            return [0]  # They can't be distracted because our workers dead

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
