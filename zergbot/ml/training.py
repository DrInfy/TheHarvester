import os

import sc2
from sc2 import Race, Difficulty, Result, AIBuild
from sc2.player import Bot, Computer
from zergbot.ml.agents import ActorCriticModel
from zergbot.theharvester import HarvesterBot

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import numpy as np
from queue import Queue

import tensorflow as tf

tf.enable_eager_execution()

LEARNING_RATE = 0.001
SAVE_DIR = "./data/"
MODEL_FILE_NAME = 'model.h5'
MAX_EPS = 10
GAMMA = 0.99

class MasterAgent():
    def __init__(self):
        self.save_dir = SAVE_DIR
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.opt = tf.train.AdamOptimizer(LEARNING_RATE, use_locking=True)

        # todo: could make this more elegant, rather than hardcoding
        self.state_size = 3
        self.action_size = 2
        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

        model_file_path = os.path.join(SAVE_DIR, MODEL_FILE_NAME)
        if os.path.isfile(model_file_path):
            self.global_model.load_weights(model_file_path)

    def train(self):
        res_queue = Queue()

        workers = [Worker(self.state_size,
                          self.action_size,
                          self.global_model,
                          self.opt, res_queue,
                          i,
                          save_dir=self.save_dir) for i in range(1)]  # todo: multiprocessing.cpu_count()

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.run()
            # worker.start() # todo: fix threading

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        # [w.join() for w in workers] # todo: fix threading


class TrainingBot(HarvesterBot):
    def __init__(self, state_size, action_size, build: str = "default"):
        super().__init__(state_size, action_size, build)

    async def on_step(self, iteration):
        return await super().on_step(iteration)

    async def on_end(self, game_result: Result):
        await super().on_end(game_result)


# class Worker(threading.Thread):
class Worker:
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.worker_idx = idx
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):

        total_step = 1
        # mem = Memory()
        while Worker.global_episode < MAX_EPS:
            bot1 = Bot(Race.Zerg, TrainingBot(3, 2))  # todo: state_size, action_size are hardcoded

            if not os.path.exists('replays'):
                os.makedirs('replays')
            sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
                bot1,
                # Computer(Race.Terran, Difficulty.VeryHard),
                Computer(Race.Terran, Difficulty.VeryHard, AIBuild.Macro)
            ], realtime=False,
                save_replay_as = f'replays/{Worker.global_episode}.SC2Replay')

            self.ep_loss = 0

            # Calculate gradient wrt to local model. We do so by tracking the
            # variables involved in computing the loss by using tf.GradientTape
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(bot1, GAMMA)
            self.ep_loss += total_loss
            # Calculate local gradients
            grads = tape.gradient(total_loss, bot1.ai.agent.local_model.trainable_weights)
            # Push local gradients to global model
            self.opt.apply_gradients(zip(grads,
                                         self.global_model.trainable_weights))
            # Update local model with new weights
            bot1.ai.agent.local_model.set_weights(self.global_model.get_weights())

            with Worker.save_lock:
                print("Saving best model to {}, "
                      "episode score: {}".format(self.save_dir, bot1.ai.agent.ep_reward))
                self.global_model.save_weights(
                    os.path.join(self.save_dir, MODEL_FILE_NAME)
                )
                # Worker.best_score = ep_reward
            Worker.global_episode += 1

            # time_count += 1
            # current_state = new_state
            # total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     # done,
                     # new_state,
                     bot,
                     # memory,
                     gamma=0.99):
        # if done:
        reward_sum = 0.  # terminal
        # else:
        #     reward_sum = self.local_model(
        #         tf.convert_to_tensor(new_state[None, :],
        #                              dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in bot.ai.agent.mem.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = bot.ai.agent.local_model(
            tf.convert_to_tensor(np.vstack(bot.ai.agent.mem.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=bot.ai.agent.mem.actions,
                                                                     logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

