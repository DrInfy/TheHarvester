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

    rewards = {}
    loss = {}

    for line in lines:
        episode = re.search('Episode: (.*?) ', line, re.IGNORECASE).group(1)
        episode = int(episode)

        reward = re.search(' Episode Reward: (.*?) ', line, re.IGNORECASE).group(1)
        rewards[episode] = int(reward)

        logged_loss = re.search(' Loss: (.*?) ', line, re.IGNORECASE).group(1)
        loss[episode] = float(logged_loss)

    # return sorted by episode
    rewards_ordered = collections.OrderedDict(sorted(rewards.items()))
    loss_ordered = collections.OrderedDict(sorted(loss.items()))
    return list(rewards_ordered.keys()),\
           list(rewards_ordered.values()), \
           list(loss_ordered.values())


def main():
    episodes, rewards, loss = get_moving_average(args.model)
    save_plot(args.model, episodes, loss, rewards)


def save_plot(model_name, episodes, loss, rewards):
    SCATTER_SIZE = 1
    LINE_WIDTH = 1.0

    model_paths = ModelPaths(model_name)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')

    ax1.set_ylabel('Reward')
    p1 = ax1.scatter(episodes, rewards, color='red', label="Episode Rewards", s=SCATTER_SIZE)
    rewards_mean_50 = pd.Series(rewards).rolling(window=10).mean()
    p2 = ax1.plot(episodes, rewards_mean_50, color='orange', label="Episode Rewards Mean 50", linestyle="-",
                  linewidth=LINE_WIDTH)[0]
    rewards_mean_200 = pd.Series(rewards).rolling(window=50).mean()
    p3 = ax1.plot(episodes, rewards_mean_200, color='yellow', label="Episode Rewards Mean 200", linestyle="--",
                  linewidth=LINE_WIDTH)[0]

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Loss')  # we already handled the x-label with ax1
    p4 = ax2.scatter(episodes, loss, label="Episode Loss", s=SCATTER_SIZE)
    loss_mean_50 = pd.Series(loss).rolling(window=10).mean()
    p5 = ax2.plot(episodes, loss_mean_50, color='green', label="Episode Loss Mean 50", linestyle="-",
                  linewidth=LINE_WIDTH)[0]
    loss_mean_200 = pd.Series(loss).rolling(window=50).mean()
    p6 = ax2.plot(episodes, loss_mean_200, color='purple', label="Episode Loss Mean 200", linestyle="--",
                  linewidth=LINE_WIDTH)[0]
    # ax2.legend(loc='center left')
    # ax2.tick_params(axis='y')
    plt.legend([p1, p2, p3, p4, p5, p6],
               ["Episode Rewards", "Episode Rewards Mean 50", "Episode Rewards Mean 200",
                "Episode Loss", "Episode Loss Mean 50", "Episode Loss Mean 200", ])
    plt.savefig(model_paths.PLOT_FILE_PATH)
    plt.show()


if __name__ == "__main__":
    main()
