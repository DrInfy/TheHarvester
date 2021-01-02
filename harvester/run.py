import random

from ladder import run_ladder_game

from sc2 import Race
from sc2.player import Bot
from harvester.theharvester import HarvesterBot

zerg_bot = Bot(Race.Zerg, HarvesterBot("scriptonly", "default"))


def main():
    # Ladder game started by LadderManager
    print("Starting ladder game...")
    result, opponentid = run_ladder_game(zerg_bot)
    print(result, " against opponent ", opponentid)


if __name__ == "__main__":
    main()
