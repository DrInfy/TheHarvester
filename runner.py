import itertools
import os
import random
import subprocess
import platform

import argparse
from typing import List, Optional

STOP_FILE: str = "runner-stop.txt"

maps = [
    # AiArena season 2
    "DeathAuraLE",
    "EternalEmpireLE",
    "EverDreamLE",
    "GoldenWallLE",
    "IceandChromeLE",
    "PillarsofgoldLE",
    "SubmarineLE",
]

# region opponents

zerg_opponents = [
    "randomzerg",
    "ai.zerg.vision",
    "ai.zerg.insane",
]


terran_opponents = [

    "randomterran",
    "ai.terran.vision",
    "ai.terran.insane",
]

protoss_opponents = [

    "randomprotoss",
    "ai.protoss.vision",
    "ai.protoss.insane",
]

other_opponents = [
    # "ai",
    "meepmeep",
    "workerrush",
]

ai_opponents = [
    "ai.protoss.vision",
    "ai.protoss.insane",
    "ai.terran.vision",
    "ai.terran.insane",
    "ai.zerg.vision",
    "ai.zerg.insane",
]

dummies = [
    # Protoss
    "4gate",
    "adept",
    "cannonrush",
    "disruptor",
    "dt",
    "robo",
    "stalker",
    "voidray",
    "zealot",
    "tempest",
    # Zerg
    "12pool",
    "200roach",
    "hydra",
    "lings",
    "macro",
    "mutalisk",
    "workerrush",
    "lurker",
    "roachburrow",
    # Terran
    "banshee",
    "bc",
    "bio",
    "cyclone",
    "marine",
    "oldrusty",
    "tank",
    "terranturtle",
    "saferaven",
]

# endregion


def main():
    parser = argparse.ArgumentParser(description="Run bot games")
    parser.add_argument("-p1", help=f"Bot name.", default="harvester")
    parser.add_argument("-p2", help=f"Bot 2 name.")
    parser.add_argument(
        "-dr", "--dry-run", help="Print commands to execute but do not launch any games.", action="store_true"
    )
    parser.add_argument("-z", "--zerg", help="Use only Zerg opponents.", action="store_true")
    parser.add_argument("-t", "--terran", help="Use only Terran opponents.", action="store_true")
    parser.add_argument("-to", "--timeout", help="Timeout in seconds.")
    parser.add_argument("-p", "--protoss", help="Use only Protoss opponents.", action="store_true")
    parser.add_argument(
        "-r", "--rounds", help="Number of rounds to play with each map and opponent combination", type=int, default=1000
    )
    parser.add_argument("--noai", help="remove in-game AI from the enemy list", action="store_true")
    parser.add_argument("--port", help="starting port to use, i.e. 10 would result in ports 10-17 being used to play.")
    args = parser.parse_args()

    run_games(args)


def run_games(args):
    players: List[str] = args.p1.split(",")
    dry_run: bool = args.dry_run
    if args.timeout:
        timeout: Optional[int] = int(args.timeout)
    else:
        timeout = None
    if dry_run:
        rounds: int = 1
        print(f"Dry run detected. Using {rounds} rounds.")
    else:
        rounds: int = args.rounds
    if args.p2:
        if args.p2 == "dummies":
            opponents: List[str] = dummies
        else:
            opponents: List[str] = args.p2.split(",")
    else:
        opponents: List[str] = get_opponents(args)

    all_games = list(itertools.product(players, opponents, maps))
    random.shuffle(all_games)

    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    for i in range(0, rounds):
        for game in all_games:
            if os.path.isfile(STOP_FILE):
                print(f"Exiting runner... {STOP_FILE} found.")
                exit(0)

            player1 = game[0]
            opponent = game[1]
            map_name = game[2]

            if platform.system() == "Linux":
                cmd = "python3.7"
            else:
                cmd = "python"

            full_command = f"{cmd} run_custom.py --map {map_name} -p1 {player1} -p2 {opponent} --release -raw"
            if args.port:
                full_command += f" --port {args.port}"

            if dry_run:
                print(full_command)
            else:
                print("starting process")
                print(full_command)
                try:
                    subprocess.call(full_command.split(" "), timeout=timeout)
                except:
                    print("An timeout exception occurred")
                print("process ended")


def get_opponents(args) -> List[str]:
    only_zerg: bool = args.zerg
    only_terran: bool = args.terran
    only_protoss: bool = args.protoss

    opponents: List = list()

    if only_zerg:
        opponents.extend(zerg_opponents)
    elif only_terran:
        opponents.extend(terran_opponents)
    elif only_protoss:
        opponents.extend(protoss_opponents)
    else:
        opponents.extend(zerg_opponents)
        opponents.extend(terran_opponents)
        opponents.extend(protoss_opponents)
        opponents.extend(other_opponents)

    if args.noai:
        for opponent in ai_opponents:
            if opponent in opponents:
                opponents.remove(opponent)

    return opponents


if __name__ == "__main__":
    main()
