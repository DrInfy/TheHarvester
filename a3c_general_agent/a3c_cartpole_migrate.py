import threading
from typing import List, Union

from numpy.core.multiarray import ndarray

from common import *
from tactics.ml.agents import BaseMLAgent


class A3CAgent(BaseMLAgent):
    def __init__(self, state_size: int, action_size: int, global_model: ActorCriticModel, opt, update_freq: int, agent_id: int, model_file_path):
        super().__init__(state_size, action_size)
        self.global_model: ActorCriticModel = global_model
        self.opt = opt
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

    def choose_action(self, state: ndarray, reward: float) -> int:
        logits, _ = self.local_model(
            tf.convert_to_tensor(state[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        self.selected_action = np.random.choice(self.action_size, p=probs.numpy()[0])

        return self.selected_action

    def post_step(self, action, current_state, done, new_state, reward):
        if done:
            reward = -1
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
            # Push local gradients to global model
            self.opt.apply_gradients(zip(grads,
                                         self.global_model.trainable_weights))
            # Update local model with new weights
            self.local_model.set_weights(self.global_model.get_weights())

            self.mem.clear()
            self.time_count = 0

            if done:  # done and print information
                Worker.global_moving_average_reward = \
                    record(Worker.global_episode, self.ep_reward, self.agent_id,
                           Worker.global_moving_average_reward,
                           self.ep_loss, self.ep_steps)
                # We must use a lock to save our model and to print to prevent data races.
                with Worker.save_lock:
                    if self.ep_reward > Worker.best_score:
                        print("Saving best model to {}, "
                              "episode score: {}".format(self.model_file_path, self.ep_reward))
                        self.global_model.save_weights(self.model_file_path)
                        Worker.best_score = self.ep_reward
                Worker.global_episode += 1
        self.ep_steps += 1

        self.time_count += 1

    def on_end(self, state: List[Union[float, int]], reward: float):
        pass


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
        self.agent = A3CAgent(state_size, action_size, global_model, opt, args.update_freq, idx,
                              os.path.join(save_dir, 'model_{}.h5'.format(game_name)))
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir

    def run(self):
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            self.agent.on_start(current_state)
            self.ep_loss = 0

            done = False
            while not done:
                action = self.agent.choose_action(current_state, 0)
                new_state, reward, done, _ = self.env.step(action)
                self.agent.post_step(action, current_state, done, new_state, reward)
                current_state = new_state


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
