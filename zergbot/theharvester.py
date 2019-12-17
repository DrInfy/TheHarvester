from typing import Dict, Callable

from sc2 import UnitTypeId, Result
from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder
from sharpy.plans import Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.zerg import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.zerg import *
from zergbot.ml.agents import *

agents: Dict[str, Callable[[int, int], BaseMLAgent]] = {
    "learning": lambda s, a: A3CAgent(s, a),
    "learning2": lambda s, a: A3CAgent(s, a),
    "random": lambda s, a: RandomAgent(s, a),
    "scripted": lambda s, a: SemiScriptedAgent(s, a)
}


class HarvesterBot(KnowledgeBot):
    agent: BaseMLAgent

    def __init__(self, build: str = "learning"):
        super().__init__("Harvester")
        if build not in agents:
            raise ValueError(f'{build} does not exist')
        self.build_text = build
        self.distribute = None
        self.execute_func = None
        self.conceded = False
        self.next_action = 0
        # todo: build proper environment that gives state_size, action_size based on what the bot can do
        self.initialize_agent(3, 2)

    def initialize_agent(self, state_size: int, action_size: int):
        self.agent = agents[self.build_text](state_size, action_size)

    async def create_plan(self) -> BuildOrder:
        self.knowledge.data_manager.set_build(self.build_text)
        self.knowledge.print(self.build_text, "Build")

        economy = Step(lambda k: self.next_action == 0, SequentialList([
            ZergUnit(UnitTypeId.DRONE, 15),
            ActExpand(2),
            ZergUnit(UnitTypeId.DRONE, 24),
            ActExpand(3),
            ZergUnit(UnitTypeId.DRONE, 38),
            ActExpand(4),
            ZergUnit(UnitTypeId.DRONE, 50),
        ]))

        units = Step(lambda k: self.next_action == 1, SequentialList([
            ActBuilding(UnitTypeId.SPAWNINGPOOL),
            BuildOrder([
                Step(RequiredMinerals(500), ActExpand(4)),
                Step(None, ZergUnit(UnitTypeId.QUEEN, 5), skip_until=lambda k: self.minerals > 150),
                ZergUnit(UnitTypeId.ZERGLING, 400),
            ])
        ]))

        tactics = SequentialList([
            PlanDistributeWorkers(),
            WorkerScout(),
            SpreadCreep(),
            InjectLarva(),
            PlanHeatOverseer(),

            PlanWorkerOnlyDefense(),
            PlanZoneDefense(),
            PlanZoneGather(),
            PlanZoneAttack(),
            PlanFinishEnemy(),
        ])

        return BuildOrder([
            AutoOverLord(),
            economy,
            units,
            tactics
        ])

    async def on_step(self, iteration):
        self.next_action = self.agent.choose_action([self.time, self.supply_workers, self.supply_army])

        # todo: turn off for ladder.
        if self.next_action == 0:
            self.client.debug_text_screen("ECON", (0.01, 0.01), (0, 255, 0), 16)
        else:
            self.client.debug_text_screen("ARMY", (0.01, 0.01), (255, 0, 0), 16)

        if not self.conceded and self.knowledge.game_analyzer.bean_predicting_defeat_for > 5:
            await self.chat_send("pineapple")
            self.conceded = True
        return await super().on_step(iteration)

    async def on_end(self, game_result: Result):
        self.agent.on_end(game_result)
        await super().on_end(game_result)
