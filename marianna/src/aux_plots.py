import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statistics import mean

from matplotlib.ticker import MaxNLocator

FIGSIZE = (10, 7)
DPI = 80

def plot_scores(scores):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    l = len(scores)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlim(0.9, l + 0.1)
    ax.set_ylim(min(scores) - 1, max(scores) + 1)
    ax.plot(range(1, l + 1), scores)

    plt.xlabel("Episode")
    plt.ylabel("Final Score")

def plot_rewards(total_rewards):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    l = len(total_rewards)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlim(0.9, l + 0.1)
    ax.set_ylim(min(total_rewards) - 1, max(total_rewards) + 1)
    ax.plot(range(1, l + 1), total_rewards, color='red')

    plt.xlabel("Episode")
    plt.ylabel("Final Reward")

def plot_scores_mean(scores, y_tick_size=1):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    l = len(scores)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlim(0.9, l + 0.1)
    ax.set_ylim(min(scores) - 1, max(scores) + 1)
    ax.set_yticks(list(np.round((np.arange(min(scores)-1, max(scores)+y_tick_size,y_tick_size)),1)))
    ax.plot(range(1, l + 1), scores, color='dodgerblue', label='Original', alpha=0.7)

    blocks = len(scores)//100
    mean_scores = []

    for b in range(blocks):
      value = mean(scores[b*100:(b+1)*100])
      mean_scores.append(value)
    
    ax.plot(range(50, l+50, 100), mean_scores, color='darkred', label='Mean', alpha=1)

    plt.legend(loc='lower right')
    plt.xlabel("Episode")
    plt.ylabel("Final Score")

def plot_rewards_mean(total_rewards, y_tick_size=100):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    l = len(total_rewards)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlim(0.9, l + 0.1)
    ax.set_ylim(min(total_rewards) - 1, max(total_rewards) + 1)
    ax.set_yticks(list(np.round((np.arange(min(total_rewards)-1, max(total_rewards)+y_tick_size,y_tick_size)),1)))
    ax.plot(range(1, l + 1), total_rewards, color='red', label='Original', alpha=0.8)

    blocks = len(total_rewards)//100
    mean_total_rewards = []

    for b in range(blocks):
      value = mean(total_rewards[b*100:(b+1)*100])
      mean_total_rewards.append(value)

    ax.plot(range(50, l+50, 100), mean_total_rewards, color='darkblue', label='Mean', alpha=1)

    plt.legend(loc='lower right')
    plt.xlabel("Episode")
    plt.ylabel("Final Reward")