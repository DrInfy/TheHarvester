import pickle
import gzip
import os
from os.path import isfile, join
from typing import Callable, List, Tuple, Union

from filelock.filelock import FileLock
from tactics.ml.agents import A3CAgent
from tactics.ml.agents.a3c_agent import record

import tensorflow as tf


def run_learning(path: str, agent: "RecordLearner", log: Callable[[str], None]) -> None:
    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
    count = len(onlyfiles)
    log(f"{count} files found in {path}")

    index = 0
    for file in onlyfiles:
        try:
            agent.learn_from_file(os.path.join(path, file))
        except:
            log(f"Failed on file {file}")
            raise

        index += 1
        if index % 100 == 0:
            percentage = "{:.1f}".format(100 * index / count)
            log(f"Progress: {index} / {count} {percentage } %")

    log(f"Learning done, now saving model.")
    agent.save_current_model()
    log(f"Model saved.")


class RecordLearner(A3CAgent):
    def __init__(
        self,
        env_name: str,
        state_size,
        action_size,
        learning_rate=0.010,
        gamma=0.995,
        start_temperature=100,
        temperature_episodes=10000,
        log_print: Callable[[str], None] = print,
        mask: List[Tuple[int, float]] = None,
    ):
        super().__init__(
            env_name,
            state_size,
            action_size,
            learning_rate,
            gamma,
            start_temperature,
            temperature_episodes,
            log_print,
            mask,
        )
        self.merge_master = False
        self.save_learning_data = False
        self.checkpoint_interval = 0

        with FileLock(self.MODEL_FILE_LOCK_PATH):
            self.optimizer: tf.keras.optimizers.Optimizer
            # with open(self.OPTIMIZER_FILE_PATH, "rb") as f:
            with gzip.open(self.OPTIMIZER_FILE_PATH, "rb") as f:
                self.optimizer = pickle.load(f)

    def learn_from_file(self, path: str):
        self.mem.load(path)
        self.on_end(self.mem.data.states[-1], self.mem.data.rewards[-1])

    def on_end(self, state: List[Union[float, int]], reward: float):
        grads = self.calc_gradients(reward, state)
        self.optimizer.apply_gradients(zip(grads, self.local_model.trainable_weights))
        self.episode += 1
        self.reset()

    def save_current_model(self):
        self.print("Saving current model")

        with FileLock(self.MODEL_FILE_LOCK_PATH):
            global_records: dict
            with open(self.GLOBAL_RECORDS_FILE_PATH, "rb") as f:
                global_records = pickle.load(f)

            self.local_model.save_weights(self.MODEL_FILE_PATH)

            # Save optimizer weights.
            with gzip.GzipFile(self.OPTIMIZER_FILE_PATH, "wb") as f:
                pickle.dump(self.optimizer, f)

            # Worker.global_episode += 1
            global_records["global_episode"] = self.episode

            # Save global records
            with open(self.GLOBAL_RECORDS_FILE_PATH, "wb") as f:
                pickle.dump(global_records, f)

        self.print("Global model saved")
