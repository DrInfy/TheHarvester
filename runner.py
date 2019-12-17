import itertools
import os
import random
import subprocess
import platform

import argparse
from typing import List

STOP_FILE: str = "runner-stop.txt"

maps = [
    "AcropolisLE",
    "DiscoBloodbathLE",
    "EphemeronLE",
    "ThunderbirdLE",
    "TritonLE",
    "WintersGateLE",
    "WorldofSleepersLE",
]

# region opponents

zerg_opponents = [
    "lings",
    "mutalisk",
    "hydra",
    "200roach",
    "roachrush",
    "macro",

    "randomzerg",

    "ai.zerg.vision",
    "ai.zerg.insane",
]

terran_opponents = [
    "marine",
    "tank",
    "cyclone",
    "bc",
    "bio",
    "banshee",
    "randomterran",

    "ai.terran.vision",
    "ai.terran.insane",
]

protoss_opponents = [
    "zealot",
    "adept",
    "stalker",
    "4gate",
    "voidray",
    "robo",
    "cannonrush",

    "randomprotoss",
    "edge",

    "ai.protoss.vision",
    "ai.protoss.insane",
]

# endregion


def main():
    parser = argparse.ArgumentParser(
        description="Run bot games"
    )
    parser.add_argument("-p1", help=f"Bot name.", default="harvester")
    parser.add_argument("-dr", "--dry-run", help="Print commands to execute but do not launch any games.", action="store_true")
    parser.add_argument("-o", "--opponent", help="Use only this opponent.")
    parser.add_argument("-z", "--zerg", help="Use only Zerg opponents.", action="store_true")
    parser.add_argument("-t", "--terran", help="Use only Terran opponents.", action="store_true")
    parser.add_argument("-p", "--protoss", help="Use only Protoss opponents.", action="store_true")
    parser.add_argument("-r", "--rounds", help="Number of rounds to play with each map and opponent combination", type=int, default=100)

    args = parser.parse_args()

    run_games(args)


def run_games(args):
    player1: str = args.p1
    dry_run: bool = args.dry_run

    if dry_run:
        rounds: int = 1
        print(f"Dry run detected. Using {rounds} rounds.")
    else:
        rounds: int = args.rounds

    opponents = get_opponents(args)

    all_games = list(itertools.product(opponents, maps))
    random.shuffle(all_games)

    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    for i in range(0, rounds):
        for game in all_games:
            if os.path.isfile(STOP_FILE):
                print(f"Exiting runner... {STOP_FILE} found.")
                exit(0)

            opponent = game[0]
            map_name = game[1]

            if platform.system() == 'Linux':
                cmd = "python3.7"
            else:
                cmd = "python"

            full_command = f"{cmd} run_custom.py {map_name} {opponent} -release -p1 {player1} -raw"
            if dry_run:
                print(full_command)
            else:
                print("starting process")
                print(full_command)

                subprocess.call([cmd, "run_custom.py", map_name, opponent, "-release", "-p1", player1, "-raw"])

                print("process ended")


def get_opponents(args) -> List[str]:
    if args.opponent:
        return [args.opponent]

    only_zerg: bool = args.zerg
    only_terran: bool = args.terran
    only_protoss: bool = args.protoss

    opponents: List = list()

    if only_zerg:
        return zerg_opponents
    if only_terran:
        return terran_opponents
    if only_protoss:
        return protoss_opponents

    opponents.extend(zerg_opponents)
    opponents.extend(terran_opponents)
    opponents.extend(protoss_opponents)

    return opponents


if __name__ == '__main__':
    main()

