from typing import Tuple

from run_custom import setup_game
from sc2 import Race
from sc2.player import Bot
from zergbot.theharvester import HarvesterBot, MlBuild


class Sc2Env:
    """ TODO: Environment that simulates what gym has for instant replacement """
    build: MlBuild
    def __init__(self, bot_name: str, game_map: str, opponent: str):
        self.opponent = opponent
        self.game_map = game_map
        self.bot_name = bot_name
        # TODO: https://github.com/openai/gym/blob/master/gym/core.py

    def reset(self):
        harvester = HarvesterBot()
        bot1 = Bot(Race.Zerg, HarvesterBot())
        self.build = harvester.
        setup_game(True, False, bot1, self.bot_name, self.opponent, self.bot_name)

    def step(self, action: int) -> Tuple[object, float, bool, dict]:
        # TODO
        pass
