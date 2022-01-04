import os
import sub_module  # Important, do not remove!
from a3c_agent.ml_debug_worker_rush_defender import WorkerRushDefender

from harvester.theharvester import HarvesterBot
from sc2 import Race
from sc2.player import Bot

from bot_loader import CommandLineGameStarter, BotDefinitions
from version import update_version_txt


def add_definitions(definitions: BotDefinitions):
    definitions.add_bot(
        "harvesterzerg",
        lambda params: Bot(
            Race.Zerg,
            HarvesterBot(
                BotDefinitions.index_check(params, 0, "learning"),
                BotDefinitions.index_check(params, 1, "default"),
                BotDefinitions.index_check(params, 2, None),
            ),
        ),
        None,
    )

    definitions.add_bot(
        "debugmlworkerrushdefender",
        lambda params: Bot(
            Race.Zerg,
            WorkerRushDefender(),
        ),
        None,
    )

def main():
    update_version_txt()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ladder_bots_path = os.path.join("Bots")
    ladder_bots_path = os.path.join(root_dir, ladder_bots_path)
    definitions: BotDefinitions = BotDefinitions(ladder_bots_path)
    add_definitions(definitions)
    starter = CommandLineGameStarter(definitions)
    starter.play()


if __name__ == "__main__":
    main()
