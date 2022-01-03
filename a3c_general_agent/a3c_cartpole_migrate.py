
import threading
from typing import List, Union

from filelock import FileLock
from numpy.core.multiarray import ndarray

from common import *
from tactics.ml.agents import BaseMLAgent

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
                 update_freq: int,
                 agent_id: int):
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

            with FileLock(MODEL_FILE_LOCK_PATH, timeout=args.timeout):
                global_model = tf.keras.models.load_model(MODEL_FILE_PATH)

                opt = load_optimizer(OPTIMIZER_FILE_PATH, global_model.trainable_variables)
                # global_model = load_model(self.state_size, self.action_size, MODEL_FILE_PATH)
                # Push local gradients to global model
                opt.apply_gradients(zip(grads, global_model.trainable_weights))
                # Update local model with new weights
                self.local_model.set_weights(global_model.get_weights())
                global_model.save(MODEL_FILE_PATH, save_format='tf')
                save_optimizer_state(opt, OPTIMIZER_FILE_PATH)

            self.mem.clear()
            self.time_count = 0
        self.ep_steps += 1

        self.time_count += 1

    def on_end(self, state: List[Union[float, int]], reward: float):
        self.post_step(self.selected_action, self.previous_state, True, state, reward)


class MasterAgent:
    def __init__(self, game_name):
        self.game_name = game_name

        if not os.path.exists(SAVE_DIR):
            print(f"Model doesn't exist - seeding...")
            os.makedirs(SAVE_DIR)
            self.seed()

    def seed(self):
        env = gym.make(self.game_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        print(f"Seeding new model at {MODEL_FILE_PATH}")
        global_model = ActorCriticModel(state_size, action_size)
        global_model(tf.convert_to_tensor(np.random.random((1, state_size)), dtype=tf.float32))
        global_model.save(MODEL_FILE_PATH, save_format='tf')

        opt = tf.keras.optimizers.Adam(args.lr)
        init_optimizer_state(opt, global_model.trainable_variables)
        save_optimizer_state(opt, OPTIMIZER_FILE_PATH)

    def train(self, num_workers):
        workers = [Worker(i, game_name=self.game_name) for i in range(num_workers)]
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        [w.join() for w in workers]

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        with FileLock(MODEL_FILE_LOCK_PATH, timeout=99999):
            model = tf.keras.models.load_model(MODEL_FILE_PATH)
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

    def __init__(self,
                 idx,
                 game_name='CartPole-v0'):
        super(Worker, self).__init__()
        self.env = gym.make(game_name).unwrapped

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = A3CAgent(state_size, action_size,
                              args.update_freq, idx)

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
    agent = MasterAgent('CartPole-v0')
    if args.train:
        agent.train(args.workers)
    else:
        agent.play()
