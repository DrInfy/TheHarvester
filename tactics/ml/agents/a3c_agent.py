
import os
from typing import List, Union, Callable

import numpy as np
import tensorflow as tf
from filelock import FileLock, Timeout
from numpy.core.multiarray import ndarray
from tensorflow.python import keras
from tensorflow.python.keras import layers

from tactics.ml.agents import BaseMLAgent
from tactics.ml.agents.memory import Memory

from loguru import logger


class ModelPaths:
    def __init__(self, model_name):
        self.SAVE_DIR = f'./data/{model_name}'
        self.MODEL_NAME = 'model'
        self.MODEL_FILE_NAME = f'{self.MODEL_NAME}.tf'
        self.MODEL_FILE_PATH = os.path.join(self.SAVE_DIR, self.MODEL_FILE_NAME)
        self.MODEL_FILE_LOCK_PATH = f'{self.MODEL_FILE_PATH}.lock'
        self.BEST_MODEL_FILE_NAME = f'{self.MODEL_NAME}_best.tf'
        self.BEST_MODEL_FILE_PATH = os.path.join(self.SAVE_DIR, self.MODEL_FILE_NAME)
        self.BEST_MODEL_FILE_LOCK_PATH = f'{self.BEST_MODEL_FILE_PATH}.lock'
        self.OPTIMIZER_FILE_NAME = f'{self.MODEL_NAME}.opt.npy'
        self.OPTIMIZER_FILE_PATH = os.path.join(self.SAVE_DIR, self.OPTIMIZER_FILE_NAME)


def compute_loss(local_model,
                 done,
                 new_state,
                 memory,
                 gamma=0.99):
    if done:
        reward_sum = 0.  # terminal
    else:
        reward_sum = local_model(
            tf.convert_to_tensor(new_state[None, :],
                                 dtype=tf.float32))[-1].numpy()[0]

    # Get discounted rewards
    discounted_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))
    # Get our advantages
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                     dtype=tf.float32) - values
    # Value loss
    value_loss = advantage ** 2

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss


def save_optimizer_state(optimizer, file_path):
    '''
    Save keras.optimizers object state.

    Arguments:
    optimizer --- Optimizer object.
    file_path --- Path to save location.

    '''

    # save weights
    np.save(file_path, optimizer.get_weights())

    return


def load_optimizer(file_path, model_train_vars, learning_rate):
    '''
    Loads keras.optimizers object state.

    Arguments:
    file_path --- Path to save location.
    model_train_vars --- List of model variables (obtained using Model.trainable_variables)

    '''

    # create optimizer
    opt = tf.keras.optimizers.Adam(learning_rate)

    # Load optimizer weights
    opt_weights = np.load(file_path, allow_pickle=True)

    # dummy zero gradients
    zero_grads = [tf.zeros_like(w) for w in model_train_vars]
    # save current state of variables
    saved_vars = [tf.identity(w) for w in model_train_vars]

    # Apply gradients which don't do nothing with Adam
    opt.apply_gradients(zip(zero_grads, model_train_vars))

    # Reload variables
    [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]

    # Set the weights of the optimizer
    opt.set_weights(opt_weights)

    return opt


def init_optimizer_state(optimizer, model_train_vars):
    '''
    Initializes keras.optimizers object state.
    This will initialize the optimizer with dummy weights so that it matches the.

    Arguments:
    optimizer --- Optimizer object to be initialized.
    model_train_vars --- List of model variables (obtained using Model.trainable_variables)

    '''

    # dummy zero gradients
    zero_grads = [tf.zeros_like(w) for w in model_train_vars]

    # Apply gradients which don't do nothing with Adam
    optimizer.apply_gradients(zip(zero_grads, model_train_vars))

    return





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


def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.

    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    logger.warning(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    return global_ep_reward


class A3CAgent(BaseMLAgent):
    def __init__(self,
                 env_name: str,
                 state_size: int,
                 action_size: int,
                 learning_rate: float,
                 update_freq: int,
                 gamma: float,
                 model_file_lock_timeout: int,
                 shared_global_vars: dict,
                 temperature_episodes=10000,  # todo: use this for what?
                 log_print: Callable[[str], None] = print,  # todo: use this for what?
                 agent_id: int=0):
        super().__init__(state_size, action_size)
        self.local_model = ActorCriticModel(self.state_size, self.action_size)

        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.gamma = gamma
        self.model_file_lock_timeout = model_file_lock_timeout

        self.mem = Memory()
        self.time_count: int = 0
        self.ep_reward: float = 0.
        self.ep_steps: int = 0
        self.ep_loss: float = 0.0

        self.agent_id = agent_id

        self.selected_action = None
        self.previous_state = None

        self.model_paths = ModelPaths(env_name)
        self.shared_global_vars = shared_global_vars

    def on_start(self, state: List[Union[float, int]]):
        self.mem.clear()
        self.time_count = 0
        self.ep_reward = 0.
        self.ep_steps = 0
        self.ep_loss = 0.0

        self.selected_action = None
        self.previous_state = None

    def choose_action(self, state: ndarray, reward: float) -> int:

        # don't do on first step
        if self.previous_state is not None:
            self.post_step(self.selected_action, self.previous_state, False, state, reward)

        logits, _ = self.local_model(
            tf.convert_to_tensor(state[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        self.selected_action = np.random.choice(self.action_size, p=probs.numpy()[0])
        self.previous_state = state

        return self.selected_action

    def post_step(self, action, current_state, done, new_state, reward):
        # if done:
        #     reward = -1  # TODO: DONT USE THIS FOR SC2
        self.ep_reward += reward
        self.mem.store(current_state, action, reward)
        if self.time_count == self.update_freq or done:
            # Calculate gradient wrt to local model. We do so by tracking the
            # variables involved in computing the loss by using tf.GradientTape
            with tf.GradientTape() as tape:
                total_loss = compute_loss(self.local_model,
                                          done,
                                          new_state,
                                          self.mem,
                                          self.gamma)
            self.ep_loss += total_loss
            # Calculate local gradients
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)

            try:
                if done:
                    # always update if we're done.
                    timeout = 600000  # 10 minutes
                else:
                    timeout = self.model_file_lock_timeout
                with FileLock(self.model_paths.MODEL_FILE_LOCK_PATH, timeout=timeout):
                    global_model = tf.keras.models.load_model(self.model_paths.MODEL_FILE_PATH, compile=False)

                    opt = load_optimizer(self.model_paths.OPTIMIZER_FILE_PATH, global_model.trainable_variables,
                                         self.learning_rate)
                    # Push local gradients to global model
                    opt.apply_gradients(zip(grads, global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(global_model.get_weights())
                    global_model.save(self.model_paths.MODEL_FILE_PATH, save_format='tf', include_optimizer=False)
                    save_optimizer_state(opt, self.model_paths.OPTIMIZER_FILE_PATH)

                    if done:
                        self.print_episode_report(self.shared_global_vars)
            except Timeout:
                # Mid-episode updates are okay to skip in the case of a timeout,
                # but if we were done, this is a problem because we're potentially losing the experience gained
                if done:
                    logger.error("Timeout while attempting to access model for final update!")

            self.mem.clear()
            self.time_count = 0
        self.ep_steps += 1

        self.time_count += 1

    def on_end(self, state: List[Union[float, int]], reward: float):
        self.post_step(self.selected_action, self.previous_state, True, state, reward)

    def print_episode_report(self, shared_global_vars: dict):
        global_moving_average_reward = shared_global_vars['global_moving_average_reward']
        global_episode = shared_global_vars['global_episode']
        best_score = shared_global_vars['best_score']

        global_moving_average_reward.value = \
            record(global_episode.value, self.ep_reward, self.agent_id,
                   global_moving_average_reward.value,
                   self.ep_loss, self.ep_steps)
        # We must use a lock to save our model and to print to prevent data races.

        if self.ep_reward > best_score.value:
            logger.warning("Saving best model to {}, "
                  "episode score: {}".format(self.model_paths.BEST_MODEL_FILE_PATH, self.ep_reward))
            self.local_model.save_weights(self.model_paths.BEST_MODEL_FILE_PATH)
            best_score.value = self.ep_reward
        global_episode.value += 1
