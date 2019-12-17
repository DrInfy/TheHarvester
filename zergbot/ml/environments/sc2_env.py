from typing import Tuple

class Sc2Env:
    """ TODO: Environment that simulates what gym has for instant replacement """
    def __init__(self, bot_name: str, game_map: str, opponent: str):
        self.opponent = opponent
        self.game_map = game_map
        self.bot_name = bot_name
        # TODO: https://github.com/openai/gym/blob/master/gym/core.py

    def reset(self):
        # TODO
        pass

    def step(self, action: int) -> Tuple[object, float, bool, dict]:
        # TODO
        pass
