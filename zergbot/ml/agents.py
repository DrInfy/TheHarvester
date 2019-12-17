import random
from math import floor

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

from sc2 import Result


class BaseMLAgent:
    """Base machine learning agent.
    """

    def choose_action(self, state):
        """Choose and return the next action.
        """
        pass


class RandomAgent(BaseMLAgent):
    """Random Agent that takes random actions in the game.
    """

    def choose_action(self, state):
        return random.randint(0, 1)


class SemiScriptedAgent(BaseMLAgent):
    """Semi Scripted Agent that takes semi-random actions in the game until it has 50 workers,
     at which point it creates army.
    """

    def choose_action(self, state):
        if state[1] >= 50:
            return 1
        else:
            return floor(state[0] / 90) % 2


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

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.mem = Memory()
        self.ep_reward = 0.

        self.previous_action = None
        self.previous_state = None

    def choose_action(self, state):
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
        self.mem.store(self.previous_state, self.previous_action, reward)  # todo: reward currently hardcoded as zero
        # # Calculate gradient wrt to local model. We do so by tracking the
        # # variables involved in computing the loss by using tf.GradientTape
        # with tf.GradientTape() as tape:
        #     total_loss = self.compute_loss(self.mem,
        #                                    0.99 # todo: gamma hardcoded
        #                                    # args.gamma
        #                                    )
        # # self.ep_loss += total_loss
        # # Calculate local gradients
        # grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        # # Push local gradients to global model
        # self.opt.apply_gradients(zip(grads,
        #                              self.global_model.trainable_weights))
        # # Update local model with new weights
        # self.local_model.set_weights(self.global_model.get_weights())

    def compute_loss(self,
                     # done,
                     # new_state,
                     memory,
                     gamma=0.99):
        # if done:
        reward_sum = 0.  # terminal
        # else:
        #     reward_sum = self.local_model(
        #         tf.convert_to_tensor(new_state[None, :],
        #                              dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
