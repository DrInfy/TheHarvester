from abc import abstractmethod
from typing import List, Union, Tuple

from sc2 import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.managers.roles import UnitTask
from sharpy.plans import BuildOrder, StepBuildGas
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from zergbot.builds.ml_build import MlBuild
from zergbot.ml.agents import BaseMLAgent

num_distraction_workers: int = 3


class WorkerDistraction_v0(MlBuild):

    def __init__(self):
        super().__init__(2, 2, self.create_plan(), result_multiplier=10000)
        self.distraction_worker_tags: List[int] = []
        self.is_dead = False

    @property
    def state(self) -> List[Union[int, float]]:
        # try:
        workers = self.ai.workers.tags_in(self.distraction_worker_tags).sorted_by_distance_to(self.ai.enemy_start_locations[0])
        self.is_dead = len(workers) == 0
        return [len(self.ai.enemy_units.closer_than(2, workers[0].position)) > 1 if len(workers) > 0 else False,
                len(workers)]
        # except KeyError:
        #     return [False, 0]

    @property
    def score(self) -> float:
        # enemy workers not mining
        not_mining_count = len(self.ai.enemy_units.of_type(UnitTypeId.DRONE).filter(lambda unit: unit.is_attacking))
        self.reward = not_mining_count
        self.reward -= len(self.distraction_worker_tags)
        self.reward += self.action  # 1 == attacking, 0 == retreating
        return self.reward

    async def start(self, knowledge: 'Knowledge'):
        await super().start(knowledge)
        distraction_workers = self.ai.workers.closest_n_units(self.ai.enemy_start_locations[0], num_distraction_workers)
        for worker in distraction_workers:
            self.knowledge.roles.set_task(UnitTask.Scouting, worker)
            self.distraction_worker_tags.append(worker.tag)


    def get_action_name_color(self, action: int) -> Tuple[str, Tuple]:
        if self.is_dead:
            return "DEAD", (255, 255, 255)
        if action == 1:
            return ("ATTACK", (255, 0, 0))
        if action == 0:
            return ("RETREAT", (0, 255, 0))

        return super().get_action_name_color(action)

    def attack(self) -> bool:
        for worker in self.ai.workers.tags_in(self.distraction_worker_tags):
            self.do(worker.attack(self.ai.enemy_start_locations[0]))
        return True

    def retreat(self):
        for worker in self.ai.workers.tags_in(self.distraction_worker_tags):
            self.do(worker.move(self.ai.start_location))
        return True

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
                    ActCustom(lambda: self.attack() if self.action == 1 else self.retreat()),
                    PlanDistributeWorkers(),
                    PlanZoneDefense(),
                    AutoOverLord(),
                    InjectLarva(),
                    PlanZoneGather(),
                    PlanZoneAttack(10),
                    PlanFinishEnemy(),
                ])
        ]
