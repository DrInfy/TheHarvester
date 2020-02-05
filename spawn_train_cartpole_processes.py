import os
import sys

sys.path.insert(1, "sharpy-sc2")
sys.path.insert(1, os.path.join("sharpy-sc2", "python-sc2"))

import subprocess
import time
import platform

STOP_FILE: str = "runner-stop.txt"

TRAINING_SCRIPT = "./train.py"

if __name__ == '__main__':
    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    if platform.system() == 'Linux':
        cmd = "python3.7"
    else:
        cmd = "python.exe"
    start_cmds = [cmd, TRAINING_SCRIPT, "-env", "cartpole"]
    processes = [subprocess.Popen(start_cmds)
                 for i in range(15)]
    run_games = True

    while run_games:
        for index, p in enumerate(processes):
            if p.poll() is not None:
                # new processes
                processes[index] = subprocess.Popen(start_cmds)
                time.sleep(0.3)  # This is to prevent sc2 from crashing on launch.


        time.sleep(0.1)
        if os.path.isfile(STOP_FILE):
            print(f"Exiting runner... {STOP_FILE} found.")
            run_games = False
            running_game = True
            # Wait for the processes to end
            while running_game:
                running_game = False
                for index, p in enumerate(processes):
                    if p.poll() is None:
                        running_game = True

                time.sleep(0.1)
