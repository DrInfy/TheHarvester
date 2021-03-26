import pickle
import gzip
import os
from os.path import isfile, join
from typing import Callable, List, Tuple, Union

from filelock.filelock import FileLock
from tactics.ml.agents import A3CAgent
from tactics.ml.agents.a3c_agent import record
import numpy as np
import tensorflow as tf
import jsonpickle


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
            agent.save_current_model()
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
            # rebuild optimizer
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            grad_vars = self.local_model.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            # Apply gradients which don't do nothing with Adam
            self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
            # Get saved weights
            opt_weights = np.load(self.OPTIMIZER_FILE_PATH, allow_pickle=True)
            # Set the weights of the optimizer
            self.optimizer.set_weights(opt_weights)

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
            with open(self.GLOBAL_RECORDS_FILE_PATH, "r") as f:
                text = f.read()
                global_records = jsonpickle.decode(text)

            self.local_model.save_weights(self.MODEL_FILE_PATH)

            # Save optimizer weights.
            np.save(self.OPTIMIZER_FILE_PATH, self.optimizer.get_weights())

            # Worker.global_episode += 1
            global_records["global_episode"] = self.episode

            # Save global records
            with open(self.GLOBAL_RECORDS_FILE_PATH, "w") as f:
                frozen = jsonpickle.encode(global_records)
                f.write(frozen)

        self.print("Global model saved")
