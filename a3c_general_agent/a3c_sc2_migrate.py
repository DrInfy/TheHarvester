import os
from queue import Queue
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from zergbot.ml.agents import BaseMLAgent


def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.

    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    tmp = int(total_loss / float(num_steps) * 1000) / 1000
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {tmp} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward


class ActorCriticModel(keras.Model):
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
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0

    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate=0.001,
                 gamma=0.99):
        super().__init__(state_size, action_size)
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.mem = Memory()

        self.prev_action = None
        self.prev_state = None

        self.ep_reward = 0

        self.ep_steps = 0
        self.time_count = 0
        self.total_step = 0
        self.gamma = gamma

        # todo: previously these were constructed in the master agent and shared among workers
        self.opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        self.result_queue = Queue()

        self.ep_loss = 0

    def evaluate_prev_action_reward(self, reward: float):
        if self.prev_action is not None:
            self.ep_reward += reward
            self.mem.store(self.prev_state, self.prev_action, reward)

    def choose_action(self, state, reward: float):
        """Choose and return the next action.
        """
        self.evaluate_prev_action_reward(reward)

        # todo: why do we need np.array array here??
        logits, _ = self.local_model(
            tf.convert_to_tensor(np.array(state)[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)
        self.prev_action = np.random.choice(self.action_size, p=probs.numpy()[0])
        self.prev_state = state

        self.ep_steps += 1

        return self.prev_action

    def on_end(self, state: List[Union[float, int]], reward: float):
        self.evaluate_prev_action_reward(reward)
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(True,
                                           state,
                                           self.mem,
                                           self.gamma)
        self.ep_loss += total_loss
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        # Push local gradients to global model
        self.opt.apply_gradients(zip(grads,
                                     self.global_model.trainable_weights))
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())

        self.mem.clear()
        self.time_count = 0

        # if done:  # done and print information
        # Worker.global_moving_average_reward = \
        #     record(Worker.global_episode, self.ep_reward, 1, #self.worker_idx,
        #            Worker.global_moving_average_reward, self.result_queue,
        #            self.ep_loss, ep_steps)
        A3CAgent.global_moving_average_reward = \
            record(A3CAgent.global_episode, self.ep_reward, 1,  # self.worker_idx,
                   A3CAgent.global_moving_average_reward, self.result_queue,
                   self.ep_loss, self.ep_steps)
        # We must use a lock to save our model and to print to prevent data races.
        # if self.ep_reward > Worker.best_score:
        if self.ep_reward > A3CAgent.best_score:
            # with Worker.save_lock:
            print("Saving best model to {}, "
                  "episode score: {}".format('tmp', self.ep_reward))
            self.global_model.save_weights(
                os.path.join('tmp',
                             'model_{}.h5'.format('sc2'))
            )
            # Worker.best_score = ep_reward
            A3CAgent.best_score = self.ep_reward
        # Worker.global_episode += 1
        A3CAgent.global_episode += 1

        # todo: this isn't how it originally want.
        self.prev_action = None
        self.prev_state = None
        self.ep_reward = 0
        self.ep_steps = 0
        self.time_count = 0
        self.total_step = 0
        self.result_queue.empty()
        self.ep_loss = 0

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

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
