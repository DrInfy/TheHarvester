from typing import Tuple, Callable

from run_custom import setup_game
from sc2 import Race
from sc2.player import Bot
from harvester.ml.environments.base_env import BaseEnv
from harvester.theharvester import HarvesterBot, MlBuild


class Sc2Env:
    """
    Environment that simulates what gym has for instant replacement
    Should look similar to gym env in structure: https://github.com/openai/gym/blob/master/gym/core.py
    """
    def __init__(self, bot_name: str, game_map: str, opponent: str, agent: str, agent_build: str):
        self.agent = agent
        self.agent_build = agent_build
        self.opponent = opponent
        self.game_map = game_map
        self.bot_name = bot_name
        # TODO: https://github.com/openai/gym/blob/master/gym/core.py

    def run(self):
        try:
            bot1 = Bot(Race.Zerg, HarvesterBot(self.agent, self.agent_build))
            # TODO: make a version of setup_game that calls manually _host_game instead
            # TODO: currently it runs the whole game, it doesn't just set it up
            setup_game(False, False, bot1, self.bot_name, self.opponent, self.game_map)
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            pass
            # self.env.close()
