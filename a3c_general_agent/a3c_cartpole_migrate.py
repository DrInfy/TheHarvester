import threading

import matplotlib.pyplot as plt
import numpy as np

from common import *


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
                 global_model,
                 opt,
                 idx,
                 game_name='CartPole-v0',
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = compute_loss(self.local_model,
                                                  done,
                                                  new_state,
                                                  mem,
                                                  args.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward,
                                   self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1


if __name__ == '__main__':

    # __init__
    game_name = 'CartPole-v0'
    save_dir = args.save_dir
    save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = gym.make(game_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    opt = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
    print(state_size, action_size)

    global_model = ActorCriticModel(state_size, action_size)  # global network
    global_model(tf.convert_to_tensor(np.random.random((1, state_size)), dtype=tf.float32))

    # train

    workers = [Worker(state_size,
                      action_size,
                      global_model,
                      opt,
                      i, game_name=game_name,
                      save_dir=save_dir) for i in range(args.workers)]

    for i, worker in enumerate(workers):
        print("Starting worker {}".format(i))
        worker.start()
    [w.join() for w in workers]
