import os
import sys
import argparse

from zergbot.ml.agents import *

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

import tensorflow as tf

from zergbot.ml.environments.sc2_env import Sc2Env

STOP_FILE: str = "runner-stop.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run bot games"
    )
    parser.add_argument("-env", help=f"Environment name (workerdistraction, harvester, cartpole).", default="workerdistraction")
    parser.add_argument("-inst", help=f"The id assigned to this training instance.", default=0)
    parser.add_argument('-train', help=f"Whether to train the agent. If this is False the agent plays instead.", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use CPU??
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):  # use CPU instead of GPU
        while not os.path.isfile(STOP_FILE):
            # try
            # https://github.com/Dentosal/python-sc2/blob/master/examples/fastreload.py
            # or
            # https://github.com/Dentosal/python-sc2/blob/master/examples/host_external_norestart.py
            if args.env == "workerdistraction":
                env = Sc2Env("test_bot.workerdistraction",
                             "AbyssalReefLE",
                             "debugmlworkerrushdefender",
                             "learning",
                             "workerdistraction",
                             train=args.train)
            elif args.env == "harvester":
                env = Sc2Env("test_bot.default",
                             "AbyssalReefLE",
                             "harvester",
                             "learning",
                             "default")

            elif args.env == "cartpole":
                agent: BaseMLAgent = A3CAgent(args.env, 4, 2)
                from zergbot.ml.environments.cartpole_env import CartPoleEnv
                env = CartPoleEnv(agent.choose_action, agent.on_end)

            env.run()

            if not args.train:
                break

    # exit()  # for some reason pycharm hangs in the python process without this?