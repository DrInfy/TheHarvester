from sys import maxsize

import gym

from tactics.ml.agents import BaseMLAgent
from tactics.ml.environments.base_env import BaseEnv


class OpenAIGymEnv(BaseEnv):
    def __init__(self, agent: BaseMLAgent, game_name, max_steps=maxsize) -> None:
        super().__init__(agent)
        self.env = gym.make(game_name).unwrapped
        self.max_steps = max_steps

    def run(self):
        state = self.env.reset()
        self.agent.on_start(state)
        reward = 0
        done = False
        try:
            while not done and self.agent.ep_steps < self.max_steps:
                # self.env.render('rgb')
                action = self.agent.choose_action(state, reward)
                state, reward, done, _ = self.env.step(action)

            self.agent.on_end(state, reward)
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            self.env.close()
