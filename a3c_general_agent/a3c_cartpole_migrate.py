
import threading
import time
from typing import List, Union

from filelock import FileLock
from numpy.core.multiarray import ndarray

from common import *
from tactics.ml.agents import BaseMLAgent

# BEGIN SEEDING FIX - THIS HOPEFULLY FIXES AN ISSUE WITH DIFFERENT BEHAVIOUR AFTER MODEL SAVE/LOAD
seedValue=1

# import os
os.environ["PYTHONHASHSEED"]=str(seedValue)

import numpy as np
np.random.seed(seedValue)

import random
random.seed(seedValue)

import tensorflow as tf
tf.random.set_seed(seedValue)
tf.compat.v1.set_random_seed(seedValue)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
# END SEEDING FIX

# Remove warning spam
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

SAVE_DIR = f'./data/{args.model_name}'
MODEL_NAME = 'model'
MODEL_FILE_NAME = f'{MODEL_NAME}.tf'
MODEL_FILE_PATH = os.path.join(SAVE_DIR, MODEL_FILE_NAME)
MODEL_FILE_LOCK_PATH = f'{MODEL_FILE_PATH}.lock'
BEST_MODEL_FILE_NAME = f'{MODEL_NAME}_best.tf'
BEST_MODEL_FILE_PATH = os.path.join(SAVE_DIR, MODEL_FILE_NAME)
BEST_MODEL_FILE_LOCK_PATH = f'{BEST_MODEL_FILE_PATH}.lock'
OPTIMIZER_FILE_NAME = f'{MODEL_NAME}.opt.npy'
OPTIMIZER_FILE_PATH = os.path.join(SAVE_DIR, OPTIMIZER_FILE_NAME)


class A3CAgent(BaseMLAgent):
    def __init__(self, state_size: int, action_size: int,
                 # global_model: ActorCriticModel,
                 # opt,
                 update_freq: int,
                 agent_id: int, model_file_path):
        super().__init__(state_size, action_size)
        # self.global_model: ActorCriticModel = global_model
        # self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.update_freq = update_freq

        self.mem = Memory()
        self.time_count: int = 0
        self.ep_reward: float = 0.
        self.ep_steps: int = 0
        self.ep_loss: float = 0.0

        self.agent_id = agent_id
        self.model_file_path = model_file_path

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

            with FileLock(MODEL_FILE_LOCK_PATH, timeout=99999):
                global_model = tf.keras.models.load_model(MODEL_FILE_PATH)
                opt = tf.keras.optimizers.Adam(args.lr)
                load_optimizer_state(opt, OPTIMIZER_FILE_PATH, global_model.trainable_variables)
                # global_model = load_model(self.state_size, self.action_size, MODEL_FILE_PATH)
                # Push local gradients to global model
                opt.apply_gradients(zip(grads,
                                             global_model.trainable_weights))
                # Update local model with new weights
                self.local_model.set_weights(global_model.get_weights())
                # global_model.save_weights(MODEL_FILE_PATH)
                global_model.save(MODEL_FILE_PATH, save_format='tf')
                save_optimizer_state(opt, OPTIMIZER_FILE_PATH)

            self.mem.clear()
            self.time_count = 0
        self.ep_steps += 1

        self.time_count += 1

    def on_end(self, state: List[Union[float, int]], reward: float):
        self.post_step(self.selected_action, self.previous_state, True, state, reward)


class MasterAgent():
    def __init__(self):
        self.game_name = 'CartPole-v0'
        self.save_dir = SAVE_DIR
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        # self.opt = tf.keras.optimizers.Adam(args.lr)
        print(self.state_size, self.action_size)

    def seed(self):
        print(f"Seeding new model at {MODEL_FILE_PATH}")
        global_model = ActorCriticModel(self.state_size, self.action_size)
        global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        global_model.save(MODEL_FILE_PATH, save_format='tf')

        opt = tf.keras.optimizers.Adam(args.lr)
        init_optimizer_state(opt, global_model.trainable_variables)
        save_optimizer_state(opt, OPTIMIZER_FILE_PATH)

    def train(self, num_workers: int = multiprocessing.cpu_count()):
        workers = [Worker(self.state_size,
                          self.action_size,
                          # self.global_model,
                          # self.opt,
                          i, game_name=self.game_name,
                          save_dir=self.save_dir) for i in range(args.workers)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        [w.join() for w in workers]

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        # model = self.global_model
        with FileLock(MODEL_FILE_LOCK_PATH, timeout=99999):
            model = tf.keras.models.load_model(MODEL_FILE_PATH)
            # model = load_model(self.state_size, self.action_size, MODEL_FILE_PATH)
        # model_path = os.path.join(args.save_dir, 'model_{}.h5'.format(self.game_name))
        # print('Loading model from: {}'.format(model_path))
        # model.load_weights(model_path)
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


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 # global_model,
                 # opt,
                 idx,
                 game_name='CartPole-v0',
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.agent = A3CAgent(state_size, action_size,
                              # global_model,
                              # opt,
                              args.update_freq, idx,
                              os.path.join(save_dir, 'model_{}.h5'.format(game_name)))
        self.env = gym.make(game_name).unwrapped
        self.save_dir = save_dir

    def run(self):
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            self.agent.on_start(current_state)
            done = False
            reward = 0
            while not done:
                action = self.agent.choose_action(current_state, reward)
                current_state, reward, done, _ = self.env.step(action)

            self.agent.on_end(current_state, reward)

            Worker.global_moving_average_reward = \
                record(Worker.global_episode, self.agent.ep_reward, self.agent.agent_id,
                       Worker.global_moving_average_reward,
                       self.agent.ep_loss, self.agent.ep_steps)
            # We must use a lock to save our model and to print to prevent data races.

            with FileLock(BEST_MODEL_FILE_LOCK_PATH, timeout=99999):
                if self.agent.ep_reward > Worker.best_score:
                    print("Saving best model to {}, "
                          "episode score: {}".format(BEST_MODEL_FILE_PATH, self.agent.ep_reward))
                    self.agent.local_model.save_weights(BEST_MODEL_FILE_PATH)
                    Worker.best_score = self.agent.ep_reward
            Worker.global_episode += 1


if __name__ == '__main__':
    agent = MasterAgent()
    if args.seed:
        agent.seed()

    if args.train:
        agent.train()
    else:
        agent.play()
