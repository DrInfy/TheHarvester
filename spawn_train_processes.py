import os
import sys

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

import subprocess
import time

TRAINIG_SCRIPT = "./train.py"

if __name__ == '__main__':
    processes = [subprocess.Popen(["python.exe", TRAINIG_SCRIPT])
                 for i in range(3)]

    while True:
        for index, p in enumerate(processes):
            if p.poll() is not None:
                # new processes
                processes[index] = subprocess.Popen(["python.exe", TRAINIG_SCRIPT])

        time.sleep(1)
