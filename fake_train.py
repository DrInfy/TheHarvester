from typing import List, Union

from zergbot.ml.agents import *
from zergbot.ml.environments.cartpole_env import CartPoleEnv

class Runner:

    def __init__(self) -> None:
        # TODO: How to get the correct values?
        self.agent: BaseMLAgent = RandomAgent(3, 2)

    def on_step(self, state: List[Union[float, int]], reward: float) -> int:
        return self.agent.choose_action(state)

    def on_end(self, state: List[Union[float, int]], reward: float):
        pass

if __name__ == '__main__':
    runner = Runner()
    env = CartPoleEnv(runner.on_step, runner.on_end)

    for i in range(0, 10):
        env.run()
