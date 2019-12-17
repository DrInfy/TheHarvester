
import os
import sys
sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))


from zergbot.ml.training import MasterAgent


if __name__ == '__main__':
    master = MasterAgent()
    # if args.train:
    master.train()
# else:
#   master.play()
