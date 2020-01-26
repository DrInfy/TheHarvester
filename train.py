import os
import sys

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

import tensorflow as tf

from a3c_general_agent.a3c_sc2_migrate import A3CAgent
from zergbot.ml.environments.sc2_env import Sc2Env

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use CPU??
    tf.enable_eager_execution()
    with tf.device('/cpu:0'):  # use CPU instead of GPU

        env = Sc2Env("test_bot.workerdistraction",
                     "AbyssalReefLE",
                     "debugmlworkerrushdefender",
                     "learning",
                     "workerdistraction")

        # env = Sc2Env("test_bot.default",
        #              "AbyssalReefLE",
        #              "debugmlworkerrushdefender",
        #              "learning",
        #              "default")
        env.run()
