import logging
import os
import pickle
from random import randint
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from filelock.filelock import FileLock
from harvester.ml.agents import BaseMLAgent
from numpy.core.multiarray import ndarray

logger = logging.getLogger(__name__)

from harvester.ml.agents import A3CAgent


class PlayA3CAgent(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """

        logits, values = self.local_model(
            tf.convert_to_tensor(state[None, :],
                                 dtype=tf.float32))

        if self.episode < self.logit_bonus_episodes:
            logits /= 1 + self.logit_bonus * (self.logit_bonus_episodes - self.episode) / self.logit_bonus_episodes

        probs = tf.nn.softmax(logits)
        # probs = self.sample(logits, 1000)
        self.prev_action = np.random.choice(self.action_size, p=probs.numpy()[0])

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

class ArgMaxA3CAgent(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        policy, value = self.local_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        self.prev_action = np.argmax(policy)

        self.ep_steps += 1

        return self.prev_action

class ArgMaxA3CAgentPlay(A3CAgent):
    def choose_action(self, state: ndarray, reward: float) -> int:
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        policy, value = self.local_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        self.prev_action = np.argmax(policy)

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
