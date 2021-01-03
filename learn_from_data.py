import argparse
import sub_module
from loguru import logger

from harvester.theharvester import HarvesterBot
from tactics.ml.agents.record_learner import run_learning


def main():
    parser = argparse.ArgumentParser(description="Run learning from data")
    parser.add_argument("-p1", help=f"Bot name. TODO", default="harvesterzerg")
    parser.add_argument("-index", help=f"Model index.", default=None)

    parser.add_argument("-p", "--paths", help="file paths separated by commas")

    args = parser.parse_args()
    paths = str(args.paths).split(",")

    bot = HarvesterBot("datalearning", "default", args.index)
    agent_manager = bot.create_agent_manager()
    agent = agent_manager.create_agent(fake_log)

    for path in paths:
        run_learning(path, agent, logger.info)


def fake_log(text: str):
    ...


if __name__ == "__main__":
    main()
