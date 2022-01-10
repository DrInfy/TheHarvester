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

    def __init__(self, bot_name_agent_build: str, opponent_name_agent_build: str, game_map: str, shared_global_vars: dict, **kwargs):
        self.bot_name_agent_build = bot_name_agent_build
        self.opponent_name_agent_build = opponent_name_agent_build
        self.game_map = game_map
        self.shared_global_vars = shared_global_vars
        self.game_starter = self.make_game_starter(shared_global_vars)
        self.kwargs = kwargs
        # TODO: https://github.com/openai/gym/blob/master/gym/core.py

    def make_game_starter(self, shared_global_vars: dict):
        """taken from run_custom.py script"""
        root_dir = os.path.dirname(os.path.abspath(__file__))
        ladder_bots_path = os.path.join("Bots")
        ladder_bots_path = os.path.join(root_dir, ladder_bots_path)
        definitions: BotDefinitions = BotDefinitions(ladder_bots_path)
        add_definitions(definitions)
        return GameStarter(definitions, shared_global_vars)

    def run(self):
        try:
            self.game_starter.play(map_name=self.game_map,
                              player1=self.bot_name_agent_build,
                              player2=self.opponent_name_agent_build, episode=self.shared_global_vars['episode'], **self.kwargs)
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        except ConnectionResetError:
            print("ConnectionResetError received. Ignoring...")
        finally:
            pass
            # self.env.close()
