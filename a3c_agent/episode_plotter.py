import argparse
import collections
import re
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from tactics.ml.agents.a3c_agent import ModelPaths

parser = argparse.ArgumentParser(description='Run A3C algorithm on a game.')
parser.add_argument("--model", help=f"Name of model to process.", required=True)
args = parser.parse_args()

def get_moving_average(model_name: str):
    model_paths = ModelPaths(model_name)
    episode_log_file = open(model_paths.EPISODE_LOG_PATH, 'r')
    lines = episode_log_file.readlines()

    # Target format:
    # Episode: 1 | Moving Average Reward: 0 | Episode Reward: 0 | Loss: 0.0 | Steps: 231 | Worker: 0

    moving_average = {}
    loss = {}

    for line in lines:
        episode = re.search('Episode: (.*?) ', line, re.IGNORECASE).group(1)
        episode = int(episode)

        logged_moving_average = re.search(' Moving Average Reward: (.*?) ', line, re.IGNORECASE).group(1)
        moving_average[episode] = int(logged_moving_average)

        logged_loss = re.search(' Loss: (.*?) ', line, re.IGNORECASE).group(1)
        loss[episode] = float(logged_loss)

    # return sorted by episode
    moving_averged_ordered = collections.OrderedDict(sorted(moving_average.items()))
    loss_ordered = collections.OrderedDict(sorted(loss.items()))
    return list(moving_averged_ordered.keys()),\
           list(moving_averged_ordered.values()), \
           list(loss_ordered.values())


def main():
    episodes, moving_average, loss = get_moving_average(args.model)
    save_plot(args.model, episodes, loss, moving_average)


def save_plot(model_name, episodes, loss, moving_average):
    model_paths = ModelPaths(model_name)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Moving Average Reward', color=color)
    ax1.plot(episodes, moving_average, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(episodes, loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(model_paths.PLOT_FILE_PATH)


if __name__ == "__main__":
    main()