import os
import sys

from tactics.ml.agents.a3c_agent import ModelPaths, ActorCriticModel, init_optimizer_state, save_optimizer_state, \
    record, A3CAgent

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

import argparse
import multiprocessing
import time
from typing import List, Union
from sys import maxsize

import gym
import numpy as np
import tensorflow as tf
from filelock import FileLock, Timeout
from numpy.core.multiarray import ndarray
from tensorflow.python import keras
from tensorflow.python.keras import layers

from tactics.ml.agents import BaseMLAgent
from tactics.ml.environments.base_env import BaseEnv
from tactics.ml.environments.sc2_env import Sc2Env

# @formatter:off
# PURPOSE: Remove warning spam
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# @formatter:on

# Disable all GPUs. This prevents errors caused by the workers all trying to use the same GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Run A3C algorithm on a game.')
parser.add_argument("--env", help=f"Environment name (workerdistraction, harvester, OpenAIGym:<EnvironmentID>).",
                    default="OpenAIGym:CartPole-v0")
parser.add_argument('--train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=-1, type=int,
                    help='How often to update the global model. Set to -1 to disable.')
# parser.add_argument('--max-eps', default=1000, type=int,
#                     help='Global maximum number of episodes to run.')
parser.add_argument('--max-steps', default=maxsize, type=int,
                    help='Maximum number of steps to run in each episode.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--model-name', default=time.strftime("%Y%m%d-%H%M%S"), type=str,
                    help='The unique name of the model you want to load or create.')
parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int,
                    help='The number of workers to run.')
parser.add_argument('--timeout', default=999999, help='Max amount of time to wait to load the model file.')
args = parser.parse_args()

STOP_FILE: str = "worker-stop.txt"


class EnvUtils:
    state_action_size_map = {
        'workerdistraction': (0, 0),
        'harvester': (0, 0),
    }

    @staticmethod
    def is_openaigym_environment(environment_name: str) -> bool:
        return environment_name.startswith("OpenAIGym:")

    @staticmethod
    def extract_openaigym_game_name(environment_name: str) -> str:
        return environment_name.split(":", 1)[1]

    @staticmethod
    def get_env_state_action_sizes(environment_name: str) -> (int, int):
        if EnvUtils.is_openaigym_environment(environment_name):
            game_name = EnvUtils.extract_openaigym_game_name(environment_name)
            gym_env = gym.make(game_name).unwrapped
            return gym_env.observation_space.shape[0], gym_env.action_space.n
        else:
            return EnvUtils.state_action_size_map[environment_name]

    @staticmethod
    def setup_environment_name_for_training(environment_name,
                                            update_freq: int,
                                            agent_id: int,
                                            max_steps: int,
                                            model_paths: ModelPaths) -> BaseEnv:
        if environment_name == "workerdistraction":
            env = Sc2Env("harvesterzerg",
                         "Simple64",
                         "debugmlworkerrushdefender",
                         "learning",
                         "workerdistraction")
        elif environment_name == "harvester":
            env = Sc2Env("test_bot.default",
                         "AbyssalReefLE",
                         "harvester",
                         "learning",
                         "default")

        elif EnvUtils.is_openaigym_environment(environment_name):
            game_name = EnvUtils.extract_openaigym_game_name(environment_name)

            state_size, action_size = EnvUtils.get_env_state_action_sizes(environment_name)

            agent: BaseMLAgent = A3CAgent(state_size, action_size, update_freq, agent_id, model_paths)
            from tactics.ml.environments.open_ai_gym_env import OpenAIGymEnv
            env = OpenAIGymEnv(agent, game_name, max_steps)
        else:
            raise f"Environment not found: {environment_name}"

        return env


def run_worker(worker_index, environment_name, model_paths: ModelPaths,
               global_episode, global_moving_average_reward, best_score, max_steps, update_freq):
    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    env = EnvUtils.setup_environment_name_for_training(environment_name, update_freq, worker_index, max_steps,
                                                       model_paths)

    while not os.path.isfile(STOP_FILE):
        env.run()

        global_moving_average_reward.value = \
            record(global_episode.value, env.agent.ep_reward, env.agent.agent_id,
                   global_moving_average_reward.value,
                   env.agent.ep_loss, env.agent.ep_steps)
        # We must use a lock to save our model and to print to prevent data races.

        with FileLock(model_paths.BEST_MODEL_FILE_LOCK_PATH, timeout=99999):
            if env.agent.ep_reward > best_score.value:
                print("Saving best model to {}, "
                      "episode score: {}".format(model_paths.BEST_MODEL_FILE_PATH, env.agent.ep_reward))
                env.agent.local_model.save_weights(model_paths.BEST_MODEL_FILE_PATH)
                best_score.value = env.agent.ep_reward
        global_episode.value += 1

    print(f"Exiting worker... {STOP_FILE} found.")


class MasterAgent:
    def __init__(self, environment_name, model_paths: ModelPaths):
        self.environment_name = environment_name
        self.model_paths = model_paths

        if not os.path.exists(self.model_paths.SAVE_DIR):
            print(f"Model doesn't exist - seeding...")
            os.makedirs(self.model_paths.SAVE_DIR)
            self.seed()

    def seed(self):
        state_size, action_size = EnvUtils.get_env_state_action_sizes(self.environment_name)

        print(f"Seeding new model at {self.model_paths.MODEL_FILE_PATH}")
        global_model = ActorCriticModel(state_size, action_size)
        global_model(tf.convert_to_tensor(np.random.random((1, state_size)), dtype=tf.float32))
        global_model.save(self.model_paths.MODEL_FILE_PATH, save_format='tf', include_optimizer=False)

        opt = tf.keras.optimizers.Adam(args.lr)
        init_optimizer_state(opt, global_model.trainable_variables)
        save_optimizer_state(opt, self.model_paths.OPTIMIZER_FILE_PATH)

    def train(self, num_workers, global_episode, global_moving_average_reward, best_score, max_steps, update_freq):
        workers = [multiprocessing.Process(target=run_worker,
                                           args=(i, self.environment_name, self.model_paths,
                                                 global_episode, global_moving_average_reward, best_score, max_steps,
                                                 update_freq))
                   for i in range(num_workers)]
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        [w.join() for w in workers]

    def play(self):
        env = gym.make(self.environment_name).unwrapped
        state = env.reset()
        with FileLock(self.model_paths.MODEL_FILE_LOCK_PATH, timeout=99999):
            model = tf.keras.models.load_model(self.model_paths.MODEL_FILE_PATH, compile=False)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


if __name__ == '__main__':
    agent = MasterAgent(args.env, ModelPaths(args.model_name))
    if args.train:
        manager = multiprocessing.Manager()
        global_episode = manager.Value('i', 0)
        global_moving_average_reward = manager.Value('i', 0)
        best_score = manager.Value('i', -maxsize)  # start with the most negative best score possible
        agent.train(args.workers, global_episode, global_moving_average_reward, best_score, args.max_steps,
                    args.update_freq)
    else:
        agent.play()
