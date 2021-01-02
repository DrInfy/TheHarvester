import logging
from typing import List, Union

import numpy as np
import tensorflow as tf

from numpy.core.multiarray import ndarray

logger = logging.getLogger(__name__)

from tactics.ml.agents import A3CAgent


class PlayA3CAgent(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        if self.save_learning_data:
            self.evaluate_prev_action_reward(reward)

        logits, values = self.local_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

        if self.episode < self.temperature_episodes:
            logits /= (
                1 + self.start_temperature * (self.temperature_episodes - self.episode) / self.temperature_episodes
            )

        probs = tf.nn.softmax(logits)
        # probs = self.sample(logits, 1000)
        self.prev_action = np.random.choice(self.action_size, p=probs.numpy()[0])

        self.ep_steps += 1

        return self.prev_action

    def on_end(self, state: List[Union[float, int]], reward: float):
        self.save_memory()
        self.reset()


class ArgMaxA3CAgent(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        policy, value = self.local_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        self.prev_action = int(np.argmax(policy))
        self.prev_state = state
        self.ep_steps += 1

        return self.prev_action


class ArgMaxA3CAgentPlay(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        policy, value = self.local_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        self.prev_action = int(np.argmax(policy))
        self.prev_state = state
        self.ep_steps += 1

        return self.prev_action

    def on_end(self, state: List[Union[float, int]], reward: float):
        # todo: this isn't how it originally was.
        self.prev_action = None
        self.prev_state = None
        self.ep_reward = 0
        self.ep_steps = 0
        self.time_count = 0
        self.total_step = 0
        self.ep_loss = 0
