from typing import Callable
import gym
from zergbot.ml.environments.base_env import BaseEnv


class CartPoleEnv(BaseEnv):
    def __init__(self, on_step: Callable, on_end: Callable) -> None:
        super().__init__(on_step, on_end)
        self.game_name = 'CartPole-v1'
        self.env = gym.make(self.game_name)

    def run(self):
        state = self.env.reset()
        reward = 0
        done = False
        try:
            while not done:
                # self.env.render('rgb')
                action = self.on_step(state, reward)
                state, reward, done, _ = self.env.step(action)
                if done:
                    self.on_end(state, reward)
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            self.env.close()
