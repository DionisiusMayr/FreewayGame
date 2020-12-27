import statistics 
from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns

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
    
    ax.plot(s1, color='cyan', label=lab1, alpha=0.8)
    ax.plot(s2, color='magenta', label=lab2, alpha=0.5)
    ax.plot(s3, color='red', label=lab3)
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
#     plt.legend()
    
#     ############3
#     # ---------- Moving average for the scores of the agent with 2 and 3 actions ----------



# s1 = scores_3act[:4000]
# s2 = scores_2act[:4000]
# s3 = baseline[:4000]

# blocks = 4000//100
# s1_avg100 = []

# for b in range(blocks-1):
#     value = statistics.mean(s1[b*100:(b+1)*100])
#     s1_avg100.append(value)

# s2_avg100 = []
# for b in range(blocks-1):
#     value = statistics.mean(s2[b*100:(b+1)*100])
#     s2_avg100.append(value)
    
# s3_avg100 = []
# for b in range(blocks-1):
#     value = statistics.mean(s3[b*100:(b+1)*100])
#     s3_avg100.append(value)
    
# aux_plots.plot_3scores(s1_avg100, s2_avg100, s3_avg100, "3 actions (up, down or stay)", "2 actions (up or stay)", "Baseline mean score (up)")