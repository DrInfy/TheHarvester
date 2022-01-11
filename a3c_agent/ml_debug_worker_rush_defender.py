from sc2 import UnitTypeId
from sc2.position import Point2
from sharpy.knowledges import KnowledgeBot
from sharpy.managers.core.roles import UnitTask
from sharpy.plans import BuildOrder, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import AutoOverLord
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import InjectLarva


class WorkerRushDefender(KnowledgeBot):
    """Dummy bot for debugging ML learning."""

    def __init__(self):
        super().__init__(type(self).__name__)
        self.SPAWNING_POOL_OFFSET = 6

    async def create_plan(self) -> BuildOrder:
        """Build the spawning pool in a place where the distracting workers won't accidentally attack it.
        Otherwise it interferes with training"""
        if self.start_location.y - self.enemy_start_locations[0].y < 0:
            # top left spawn - build below start location
            spawning_pool_pos = Point2((self.start_location.x, self.start_location.y+self.SPAWNING_POOL_OFFSET))
        else:
            # bottom right spawn - build above start location
            spawning_pool_pos = Point2((self.start_location.x, self.start_location.y-self.SPAWNING_POOL_OFFSET))

        return BuildOrder([
            SequentialList([
                ActUnit(UnitTypeId.DRONE, UnitTypeId.LARVA, 14),
                Expand(2),
                BuildPosition(UnitTypeId.SPAWNINGPOOL, spawning_pool_pos),
                ActUnit(UnitTypeId.OVERLORD, UnitTypeId.LARVA, 2),
                ActUnit(UnitTypeId.QUEEN, UnitTypeId.HATCHERY, 1),
                ActUnit(UnitTypeId.ZERGLING, UnitTypeId.LARVA, 200),
            ]),
            SequentialList(
                [
                    DistributeWorkers(),
                    PlanZoneDefense(),
                    AutoOverLord(),
                    InjectLarva(),
                    PlanZoneGather(),
                    PlanZoneAttack(10),
                    PlanFinishEnemy(),
                ])
        ])

    async def on_step(self, iteration):
        await super().on_step(iteration)
        enemy = self.enemy_units.closer_than(15, self.start_location.position)
        if enemy:
            for worker in self.workers:
                self.knowledge.roles.set_task(UnitTask.Attacking, worker)
                worker.attack(enemy[0])
        else:
            for worker in self.workers:
                self.knowledge.roles.set_task(UnitTask.Gathering, worker)
