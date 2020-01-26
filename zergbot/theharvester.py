from typing import Dict, Callable, Union, List

from sc2 import Result
from sharpy.knowledges import KnowledgeBot
from sharpy.managers.roles import UnitTask
from sharpy.plans import BuildOrder
from zergbot.builds import *
from zergbot.builds.worker_distraction import WorkerDistraction_v0
from zergbot.ml.agents import *

agents: Dict[str, Callable[[int, int], BaseMLAgent]] = {
    "learning": lambda s, a: A3CAgent(s, a),
    "random": lambda s, a: RandomAgent(s, a),
    "scripted": lambda s, a: SemiScriptedAgent(s, a)
}

builds: Dict[str, Callable[[], MlBuild]] = {
    "default": lambda: EconLings_v0(),
    "workerdistraction": lambda: WorkerDistraction_v0()
}


class HarvesterBot(KnowledgeBot):
    agent: BaseMLAgent
    ml_build: MlBuild

    num_distraction_workers: int = 3

    def __init__(self, agent: Union[str, BaseMLAgent] = "random", build: str = "default"):
        super().__init__("Harvester")
        self.agent = agent
        if build not in builds:
            raise ValueError(f'{build} does not exist')
        self.build_text = build
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.next_action = 0
        self.initialize_agent(agent, build)

        self.distraction_worker_tags: List[int] = []

    def initialize_agent(self, agent: Union[str, BaseMLAgent], build_text):
        self.ml_build = builds[build_text]()

        if isinstance(agent, BaseMLAgent):
            self.agent = agent
        else:
            self.agent = agents[self.build_text](self.ml_build.state_size, self.ml_build.action_size)

    async def create_plan(self) -> BuildOrder:
        self.knowledge.data_manager.set_build(self.build_text)
        self.knowledge.print(self.build_text, "Build")

        return self.ml_build

    async def on_step(self, iteration):

        state = self.ml_build.state
        action = self.agent.choose_action(state, 0)
        if state[1] > 0:  # if a scouting worker is alive
            if action == 0:
                self.attack()
            else:
                self.retreat()
        else:
            self.dead()


        return await super().on_step(iteration)

    async def on_end(self, game_result: Result):
        self.ml_build.on_end(game_result)
        self.agent.on_end([self.time, self.supply_workers, self.supply_army], self.ml_build.reward)
        await super().on_end(game_result)

    async def on_start(self):
        await super().on_start()

        distraction_workers = self.workers.closest_n_units(self.enemy_start_locations[0], HarvesterBot.num_distraction_workers)
        for worker in distraction_workers:
            self.knowledge.roles.set_task(UnitTask.Scouting, worker)
            self.distraction_worker_tags.append(worker.tag)

    def attack(self):
        self.client.debug_text_screen("ATTACK", (0.01, 0.01), (255, 0, 0), 16)
        for worker in self.workers.tags_in(self.distraction_worker_tags):
            self.do(worker.attack(self.enemy_start_locations[0]))

    def retreat(self):
        self.client.debug_text_screen("RETREAT", (0.01, 0.01), (0, 255, 0), 16)
        for worker in self.workers.tags_in(self.distraction_worker_tags):
            self.do(worker.move(self.start_location))

    def dead(self):
        self.client.debug_text_screen("DEAD", (0.01, 0.01), (255, 255, 255), 16)
