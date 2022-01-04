# import gzip
# import os
# import pickle
#
# import jsonpickle


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

    # def save(self, path_name: str, file_name: str):
    #     if not os.path.exists(path_name):
    #         os.makedirs(path_name)
    #
    #     full_path = os.path.join(path_name, file_name) + ".pgz"
    #     # frozen = jsonpickle.encode(self.data)
    #
    #     try:
    #         # with open(full_path, "w") as handle:
    #         with gzip.GzipFile(full_path, "wb") as handle:
    #             pickle.dump(self.data, handle)
    #             # handle.write(frozen)
    #     except Exception as e:
    #         print(f"Data write failed: {e}")
    #
    # def load(self, file_path: str):
    #     # with open(file_path, "r") as handle:
    #     with gzip.open(file_path, "rb") as handle:
    #         # text = handle.read()
    #         self.data = pickle.load(handle)
