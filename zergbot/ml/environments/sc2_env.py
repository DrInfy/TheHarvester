from typing import Tuple

from run_custom import setup_game
from sc2 import Race
from sc2.player import Bot
from zergbot.theharvester import HarvesterBot, MlBuild


class Sc2Env:
    """ TODO: Environment that simulates what gym has for instant replacement """
    build: MlBuild
    def __init__(self, bot_name: str, game_map: str, opponent: str):
        self.opponent = opponent
        self.game_map = game_map
        self.bot_name = bot_name
        # TODO: https://github.com/openai/gym/blob/master/gym/core.py

    def reset(self):
        harvester = HarvesterBot()
        bot1 = Bot(Race.Zerg, HarvesterBot())
        self.build = harvester.ml_build
        setup_game(True, False, bot1, self.bot_name, self.opponent, self.bot_name)

    def step(self, action: int) -> Tuple[object, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        observation = self.build.state
        reward = self.build.score
        done = self.build.game_ended
        info = {}

        return observation, reward, done, info
