import statistics 
from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.ticker import MaxNLocator

FIGSIZE = (10, 6)
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

    return ax

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

    return ax

def plot_2scores(s1, s2, lab1, lab2):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(s1, color='cyan', label=lab1, alpha=0.8)
    ax.plot(s2, color='magenta', label=lab2, alpha=0.5)
    plt.legend()
    
    plt.xlabel("Episode")
    plt.ylabel("Final Score")

    return ax

def plot_3scores(s1, s2, s3, lab1, lab2, lab3):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(s1, color='cyan', label=lab1, alpha=0.5)
    ax.plot(s2, color='magenta', label=lab2, alpha=0.5)
    ax.plot(s3, color='#C02323', label=lab3, alpha=0.5)
    plt.legend()
    
    plt.xlabel("Episode")
    plt.ylabel("Final Score")

    return ax
    
def plot_3rewards(r1, r2, r3, lab1, lab2, lab3):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_xlim(0, len(r1)-1)
    ax.plot(r1, color='#5CADFF', label=lab1, alpha=0.8)
    ax.plot(r2, color='#B981EE', label=lab2, alpha=0.8)
    ax.plot(r3, color='#FF858D', label=lab3, alpha=0.8)  # ~Red color
    plt.legend()
    
    plt.xlabel("Episode")
    plt.ylabel("Final Reward")

    return ax
    
def plot_2rewards(r1, r2, lab1, lab2):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
 
    ax.plot(r1, color='#5CADFF', label=lab1, alpha=0.8)
    ax.plot(r2, color='#B981EE', label=lab2, alpha=0.8)
    plt.legend()
    
    plt.xlabel("Episode")
    plt.ylabel("Final Reward")
    
    return ax

    
def moving_average(arr, ax, label: str, color: str):
    blocks = len(arr) // 100
    avg100 = []
    x = []

    for b in range(blocks):
        value = statistics.mean(arr[b * 100:(b + 1) * 100])
        avg100.append(value)
        x.append((b + 1) * 100)

    ax.plot(x, avg100, color=color, label=label, linewidth=2)


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
