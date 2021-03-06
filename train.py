import os
import sys
import argparse

from tactics.ml.agents import *

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

import tensorflow as tf

from tactics.ml.environments.sc2_env import Sc2Env

STOP_FILE: str = "runner-stop.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run bot games"
    )
    parser.add_argument("-env", help=f"Environment name (workerdistraction, harvester, cartpole).", default="workerdistraction")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use CPU??
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):  # use CPU instead of GPU
        while not os.path.isfile(STOP_FILE):
            if args.env == "workerdistraction":
                env = Sc2Env("test_bot.workerdistraction",
                             "AbyssalReefLE",
                             "debugmlworkerrushdefender",
                             "learning",
                             "workerdistraction")
            elif args.env == "harvester":
                env = Sc2Env("test_bot.default",
                             "AbyssalReefLE",
                             "harvester",
                             "learning",
                             "default")

            elif args.env == "cartpole":
                agent: BaseMLAgent = A3CAgent(args.env, 4, 2)
                from tactics.ml.environments.cartpole_env import CartPoleEnv
                env = CartPoleEnv(agent.choose_action, agent.on_end)

            env.run()
