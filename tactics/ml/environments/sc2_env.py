import os

from bot_loader import BotDefinitions, GameStarter
from run_custom import add_definitions
from sc2 import Race
from sc2.player import Bot
from harvester.theharvester import HarvesterBot


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

    def make_game_starter(self):
        """take from run_custom.py script"""
        root_dir = os.path.dirname(os.path.abspath(__file__))
        ladder_bots_path = os.path.join("Bots")
        ladder_bots_path = os.path.join(root_dir, ladder_bots_path)
        definitions: BotDefinitions = BotDefinitions(ladder_bots_path)
        add_definitions(definitions)
        return GameStarter(definitions)

    def run(self):
        try:
            game_starter = self.make_game_starter()
            game_starter.play(map_name=self.game_map,
                              player1=f"{self.bot_name}.{self.agent}.{self.agent_build}",
                              player2=self.opponent)
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            pass
            # self.env.close()
