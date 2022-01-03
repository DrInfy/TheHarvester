import argparse
import multiprocessing
import os
import time
from typing import List, Union

# Remove warning spam
import absl.logging
import gym
import numpy as np
import tensorflow as tf
from filelock import FileLock, Timeout
from numpy.core.multiarray import ndarray
from tensorflow.python import keras
from tensorflow.python.keras import layers

from tactics.ml.agents import BaseMLAgent

absl.logging.set_verbosity(absl.logging.ERROR)

# Disable all GPUs. This prevents errors caused by the workers all trying to use the same GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Run A3C algorithm on a game.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=-1, type=int,
                    help='How often to update the global model. Set to -1 to disable.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--model-name', default=time.strftime("%Y%m%d-%H%M%S"), type=str,
                    help='The unique name of the model you want to load or create.')
parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int,
                    help='The number of workers to run.')
parser.add_argument('--timeout', default=999999, help='Max amount of time to wait to load the model file.')
args = parser.parse_args()


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


def load_model(state_size: int, action_size: int, model_file_path: str):
    model = ActorCriticModel(state_size, action_size)
    model(tf.convert_to_tensor(np.random.random((1, state_size)), dtype=tf.float32))
    model.load_weights(model_file_path)
    return model


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


def load_optimizer(file_path, model_train_vars):
    '''
    Loads keras.optimizers object state.

    Arguments:
    file_path --- Path to save location.
    model_train_vars --- List of model variables (obtained using Model.trainable_variables)

    '''

    # create optimizer
    opt = tf.keras.optimizers.Adam(args.lr)

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
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    return global_ep_reward


class A3CAgent(BaseMLAgent):
    def __init__(self, state_size: int, action_size: int,
                 update_freq: int,
                 agent_id: int,
                 model_paths: ModelPaths):
        super().__init__(state_size, action_size)
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.update_freq = update_freq

        self.mem = Memory()
        self.time_count: int = 0
        self.ep_reward: float = 0.
        self.ep_steps: int = 0
        self.ep_loss: float = 0.0

        self.agent_id = agent_id

        self.selected_action = None
        self.previous_state = None

        self.model_paths = model_paths

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
        if done:
            reward = -1  # TODO: DONT USE THIS FOR SC2
        self.ep_reward += reward
        self.mem.store(current_state, action, reward)
        if self.time_count == args.update_freq or done:
            # Calculate gradient wrt to local model. We do so by tracking the
            # variables involved in computing the loss by using tf.GradientTape
            with tf.GradientTape() as tape:
                total_loss = compute_loss(self.local_model,
                                          done,
                                          new_state,
                                          self.mem,
                                          args.gamma)
            self.ep_loss += total_loss
            # Calculate local gradients
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)

            try:
                with FileLock(self.model_paths.MODEL_FILE_LOCK_PATH, timeout=args.timeout):
                    global_model = tf.keras.models.load_model(self.model_paths.MODEL_FILE_PATH)

                    opt = load_optimizer(self.model_paths.OPTIMIZER_FILE_PATH, global_model.trainable_variables)
                    # Push local gradients to global model
                    opt.apply_gradients(zip(grads, global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(global_model.get_weights())
                    global_model.save(self.model_paths.MODEL_FILE_PATH, save_format='tf')
                    save_optimizer_state(opt, self.model_paths.OPTIMIZER_FILE_PATH)
            except Timeout:
                pass  # move onto the next episode

            self.mem.clear()
            self.time_count = 0
        self.ep_steps += 1

        self.time_count += 1

    def on_end(self, state: List[Union[float, int]], reward: float):
        self.post_step(self.selected_action, self.previous_state, True, state, reward)


STOP_FILE: str = "worker-stop.txt"


def run_worker(worker_index, game_name, model_paths: ModelPaths,
               global_episode, global_moving_average_reward, best_score):
    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    env = gym.make(game_name).unwrapped

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A3CAgent(state_size, action_size,
                     args.update_freq, worker_index, model_paths)

    run_games = True
    while run_games:
        current_state = env.reset()
        agent.on_start(current_state)
        done = False
        reward = 0
        while not done:
            action = agent.choose_action(current_state, reward)
            current_state, reward, done, _ = env.step(action)

        agent.on_end(current_state, reward)

        global_moving_average_reward.value = \
            record(global_episode.value, agent.ep_reward, agent.agent_id,
                   global_moving_average_reward.value,
                   agent.ep_loss, agent.ep_steps)
        # We must use a lock to save our model and to print to prevent data races.

        with FileLock(model_paths.BEST_MODEL_FILE_LOCK_PATH, timeout=99999):
            if agent.ep_reward > best_score.value:
                print("Saving best model to {}, "
                      "episode score: {}".format(model_paths.BEST_MODEL_FILE_PATH, agent.ep_reward))
                agent.local_model.save_weights(model_paths.BEST_MODEL_FILE_PATH)
                best_score.value = agent.ep_reward
        global_episode.value += 1

        if os.path.isfile(STOP_FILE):
            print(f"Exiting worker... {STOP_FILE} found.")
            run_games = False


class MasterAgent:
    def __init__(self, game_name, model_paths: ModelPaths):
        self.game_name = game_name
        self.model_paths = model_paths

        if not os.path.exists(self.model_paths.SAVE_DIR):
            print(f"Model doesn't exist - seeding...")
            os.makedirs(self.model_paths.SAVE_DIR)
            self.seed()

    def seed(self):
        env = gym.make(self.game_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        print(f"Seeding new model at {self.model_paths.MODEL_FILE_PATH}")
        global_model = ActorCriticModel(state_size, action_size)
        global_model(tf.convert_to_tensor(np.random.random((1, state_size)), dtype=tf.float32))
        global_model.save(self.model_paths.MODEL_FILE_PATH, save_format='tf')

        opt = tf.keras.optimizers.Adam(args.lr)
        init_optimizer_state(opt, global_model.trainable_variables)
        save_optimizer_state(opt, self.model_paths.OPTIMIZER_FILE_PATH)

    def train(self, num_workers, global_episode, global_moving_average_reward, best_score):
        workers = [multiprocessing.Process(target=run_worker,
                                           args=(i, self.game_name, self.model_paths,
                                                 global_episode, global_moving_average_reward, best_score))
                   for i in range(num_workers)]
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        [w.join() for w in workers]

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        with FileLock(self.model_paths.MODEL_FILE_LOCK_PATH, timeout=99999):
            model = tf.keras.models.load_model(self.model_paths.MODEL_FILE_PATH)
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
    agent = MasterAgent('CartPole-v0', ModelPaths(args.model_name))
    if args.train:
        manager = multiprocessing.Manager()
        global_episode = manager.Value('i', 0)
        global_moving_average_reward = manager.Value('i', 0)
        best_score = manager.Value('i', 0)
        agent.train(args.workers, global_episode, global_moving_average_reward, best_score)
    else:
        agent.play()
