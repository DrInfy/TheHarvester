import os
import sys
import time
import traceback

from harvester.builds import EconLings_v0, EconLingRoach

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

from harvester.builds.worker_distraction import WorkerDistraction_v0
from tactics.ml.agents.a3c_agent import ModelPaths, ActorCriticModel, init_optimizer_state, save_optimizer_state, \
    A3CAgent


import argparse
import multiprocessing
from sys import maxsize

import gym
import numpy as np
import tensorflow as tf
from filelock import FileLock

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
parser.add_argument("--env", help=f"Environment name (workerdistraction, harvester, OpenAIGym.<EnvironmentID>).",
                    default="OpenAIGym.CartPole-v0")
parser.add_argument('--train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=-1, type=int,
                    help='How often to update the global model. Set to -1 to disable.')
# parser.add_argument('--max-eps', default=1000, type=int,
#                     help='Global maximum number of episodes to run.')
parser.add_argument('--max-steps', default=maxsize, type=int,
                    help='Maximum number of steps to run in each episode.')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='Discount factor of rewards.')
# parser.add_argument('--model-name', default=time.strftime("%Y%m%d-%H%M%S"), type=str,
#                     help='The unique name of the model you want to load or create.')
parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int,
                    help='The number of workers to run.')
parser.add_argument('--timeout', default=999999, help='Max amount of time to wait to load the model file.')
args = parser.parse_args()

STOP_FILE: str = "worker-stop.txt"


class EnvUtils:
    state_action_size_map = {
        'workerdistraction': (WorkerDistraction_v0.STATE_SIZE, WorkerDistraction_v0.ACTION_SIZE),
        'econ_lings': (EconLings_v0.STATE_SIZE, EconLings_v0.ACTION_SIZE),
        'econ_lings_roach': (EconLingRoach.STATE_SIZE, EconLingRoach.ACTION_SIZE),
        'harvester': (0, 0),
    }

    @staticmethod
    def is_openaigym_environment(environment_name: str) -> bool:
        return environment_name.startswith("OpenAIGym.")

    @staticmethod
    def extract_openaigym_game_name(environment_name: str) -> str:
        return environment_name.split(".", 1)[1]

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
                                            learning_rate: float,
                                            update_freq: int,
                                            gamma: float,
                                            model_file_lock_timeout: int,
                                            agent_id: int,
                                            max_steps: int,
                                            shared_global_vars: dict) -> BaseEnv:
        if environment_name == "workerdistraction":
            env = Sc2Env("harvesterzerg.learning.workerdistraction",
                         "debugmlworkerrushdefender",
                         "Simple64",
                         shared_global_vars,
                         learning_rate=learning_rate,
                         update_freq=update_freq,
                         gamma = gamma,
                         model_file_lock_timeout = model_file_lock_timeout)
        elif environment_name == "econ_lings":
            env = Sc2Env("harvesterzerg.learning.econ_lings",
                         "harvesterzerg.learning.econ_lings",
                         "Simple64",
                         shared_global_vars,
                         learning_rate=learning_rate,
                         update_freq=update_freq,
                         gamma = gamma,
                         model_file_lock_timeout = model_file_lock_timeout)
        elif environment_name == "econ_lings_roach":
            env = Sc2Env("harvesterzerg.learning.econ_lings_roach",
                         "ai.zerg.veryhard",
                         "Simple128",
                         shared_global_vars,
                         learning_rate=learning_rate,
                         update_freq=update_freq,
                         gamma = gamma,
                         model_file_lock_timeout = model_file_lock_timeout)
        # elif environment_name == "harvester":
        #     env = Sc2Env("test_bot.default",
        #                  "AbyssalReefLE",
        #                  "harvester",
        #                  "learning",
        #                  "default",
        #                  shared_global_vars,
        #                  learning_rate=learning_rate,
        #                  update_freq=update_freq,
        #                  gamma = gamma,
        #                  model_file_lock_timeout = model_file_lock_timeout)

        elif EnvUtils.is_openaigym_environment(environment_name):
            game_name = EnvUtils.extract_openaigym_game_name(environment_name)

            state_size, action_size = EnvUtils.get_env_state_action_sizes(environment_name)

            agent: BaseMLAgent = A3CAgent(environment_name,
                                          state_size,
                                          action_size,
                                          learning_rate,
                                          update_freq,
                                          gamma,
                                          model_file_lock_timeout,
                                          shared_global_vars,
                                          agent_id=agent_id)
            from tactics.ml.environments.open_ai_gym_env import OpenAIGymEnv
            env = OpenAIGymEnv(agent, game_name, max_steps)
        else:
            raise f"Environment not found: {environment_name}"

        return env


def run_worker(worker_index,
               environment_name,
               learning_rate,
               update_freq,
               gamma,
               model_file_lock_timeout,
               max_steps,
               global_episode,
               global_moving_average_reward,
               best_score):
    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    shared_global_vars = {
        'global_episode': global_episode,
        'global_moving_average_reward': global_moving_average_reward,
        'best_score': best_score,
        'episode': 0,
    }

    env = EnvUtils.setup_environment_name_for_training(environment_name,
                                                       learning_rate,
                                                       update_freq,
                                                       gamma,
                                                       model_file_lock_timeout,
                                                       worker_index,
                                                       max_steps,
                                                       shared_global_vars)

    while not os.path.isfile(STOP_FILE):
        try:
            global_episode.value += 1
            episode = global_episode.value
            shared_global_vars['episode'] = episode
            env.run()
        except Exception as ex:
            print(f"Exception caught in environment run!")
            print(ex)
            traceback.print_exc()

    print(f"Exiting worker... {STOP_FILE} found.")


class MasterAgent:
    def __init__(self, environment_name):
        self.environment_name = environment_name
        self.model_paths = ModelPaths(self.environment_name)

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

    def train(self, num_workers,
              learning_rate,
              update_freq,
              gamma,
              model_file_lock_timeout,
              max_steps,
              global_episode,
              global_moving_average_reward,
              best_score):
        workers = [multiprocessing.Process(target=run_worker,
                                           args=(i,
                                                 self.environment_name,
                                                 learning_rate,
                                                 update_freq,
                                                 gamma,
                                                 model_file_lock_timeout,
                                                 max_steps,
                                                 global_episode,
                                                 global_moving_average_reward,
                                                 best_score))
                   for i in range(num_workers)]
        WORKER_START_DELAY_SECS = 2
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
            print(f"Waiting {WORKER_START_DELAY_SECS} seconds...")
            time.sleep(2)
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
    agent = MasterAgent(args.env)
    if args.train:
        manager = multiprocessing.Manager()
        global_episode = manager.Value('i', 0)
        global_moving_average_reward = manager.Value('i', 0)
        best_score = manager.Value('i', -maxsize)  # start with the most negative best score possible

        agent.train(args.workers,
                    args.lr,
                    args.update_freq,
                    args.gamma,
                    args.timeout,
                    args.max_steps,
                    global_episode,
                    global_moving_average_reward,
                    best_score)
    else:
        agent.play()
