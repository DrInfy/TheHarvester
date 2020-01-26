import os

import tensorflow as tf

from a3c_general_agent.a3c_sc2_migrate import A3CAgent
from zergbot.ml.agents import BaseMLAgent
from zergbot.ml.environments.sc2_env import Sc2Env

if __name__ == '__main__':
    # import sys
    # sys.path.append('./a3c_general_agent/')

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.enable_eager_execution()

    # agent: BaseMLAgent = A3CAgent(4, 2)
    # env = CartPoleEnv(agent.choose_action, agent.on_end)
    env = Sc2Env("test_bot.workerdistraction", "AbyssalReefLE", "debugmlworkerrushdefender", A3CAgent(2, 2), "workerdistraction")

    for i in range(0, 1000):
        env.run()
