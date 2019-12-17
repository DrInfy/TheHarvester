from typing import Union, List

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

from sc2 import Result
from zergbot.ml.agents import BaseMLAgent


class ActorCriticModel(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class A3CAgent(BaseMLAgent):
    """A3C machine learning agent.
    """

    REWARD_WIN = 1
    REWARD_LOSE = -1
    REWARD_TIE = 0

    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.mem = Memory()
        self.ep_reward = 0.

        self.previous_action = None
        self.previous_state = None

    def choose_action(self, state: List[Union[float, int]]):
        if self.previous_state is not None and self.previous_action is not None:
            # self.ep_reward += reward
            self.mem.store(self.previous_state, self.previous_action, 0)  # todo: reward currently hardcoded as zero

        logits, _ = self.local_model(
            tf.convert_to_tensor([state],  # [[0, 1, 2]]
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        action = np.random.choice(self.action_size, p=probs.numpy()[0])

        self.previous_state = state
        self.previous_action = action
        return action

    def on_end(self, game_result: Result):
        reward = self.REWARD_TIE
        if game_result == Result.Victory:
            reward = self.REWARD_WIN
        elif game_result == Result.Defeat:
            reward = self.REWARD_LOSE
        self.mem.store(self.previous_state, self.previous_action, reward)
        self.ep_reward = reward
