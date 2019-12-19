from typing import Tuple

from run_custom import setup_game
from sc2 import Race
from sc2.player import Bot
from zergbot.theharvester import HarvesterBot, MlBuild


class Sc2Env:
    """
    Environment that simulates what gym has for instant replacement
    Should look similar to gym env in structure: https://github.com/openai/gym/blob/master/gym/core.py
    """
    build: MlBuild
    def __init__(self, bot_name: str, game_map: str, opponent: str):
        self.opponent = opponent
        self.game_map = game_map
        self.bot_name = bot_name
        # TODO: https://github.com/openai/gym/blob/master/gym/core.py

    def reset(self) -> object:
        """Returns:
            observation (object): agent's observation of the current environment"""
        harvester = HarvesterBot()
        bot1 = Bot(Race.Zerg, HarvesterBot())
        self.build = harvester.ml_build
        # TODO: make a version of setup_game that calls manually _host_game instead
        setup_game(True, False, bot1, self.bot_name, self.opponent, self.bot_name)
        # TODO: wait until first on_step, then return current state
        return None

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

        # TODO: Actions are being used on old observation data
        self.build.action = action

        # TODO: Take a step here

        observation = self.build.state
        reward = self.build.score
        done = self.build.game_ended
        info = {}

        return observation, reward, done, info
