# Script for creating Ladder Manager compatible Zip archives.
# 7-Zip command line documentation https://sevenzip.osdn.jp/chm/cmdline/

import os
import argparse
import sys
sys.path.insert(1, "sharpy-sc2")
from dummy_ladder_zip import create_ladder_zip, LadderZip
from version import update_version_txt

root_dir = os.path.dirname(os.path.abspath(__file__))

# Files or folders common to all bots.
common = [
    (os.path.join("sharpy-sc2", "jsonpickle"), "jsonpickle"),
    (os.path.join("sharpy-sc2", "sharpy"), "sharpy"),
    (os.path.join("sharpy-sc2", "python-sc2", "sc2"), "sc2"),
    (os.path.join("sharpy-sc2", "sc2pathlibp"), "sc2pathlibp"),
    ("requirements.txt", None),
    ("version.txt", None),
    (os.path.join("sharpy-sc2", "config.py"), "config.py"),
    ("config.ini", None),
    (os.path.join("sharpy-sc2", "ladder.py"), "ladder.py"),
    ("ladderbots.json", None),
]

# Files or folders to be ignored from the archive.
ignored = [
    "__pycache__",
]

zerg_zip = LadderZip("TheHarvester", "Zerg", [
    ("harvester", None),
    (os.path.join("zergbot", "run.py"), "run.py"),
], common)


zip_types = {
    "harvester": zerg_zip,

    # All
    "all": None
}

def get_archive(bot_name: str) -> LadderZip:
    bot_name = bot_name.lower()
    return zip_types.get(bot_name)

def main():
    zip_keys = list(zip_types.keys())
    parser = argparse.ArgumentParser(
        description="Create a Ladder Manager ready zip archive for SC2 AI, AI Arena, Probots, ..."
    )
    parser.add_argument("-n", "--name", help=f"Bot name: {zip_keys}.")
    parser.add_argument("-e", "--exe", help="Also make executable (Requires pyinstaller)", action="store_true")
    args = parser.parse_args()

    bot_name = args.name

    if not os.path.exists('dummy'):
        os.mkdir('dummy')

    if bot_name == "all" or not bot_name:
        zip_keys.remove("all")
        for key in zip_keys:
            create_ladder_zip(get_archive(key), args.exe)
    else:
        if bot_name not in zip_keys:
            raise ValueError(f'Unknown bot: {bot_name}, allowed values are: {zip_keys}')

        create_ladder_zip(get_archive(bot_name), args.exe)


if __name__ == "__main__":
    main()
