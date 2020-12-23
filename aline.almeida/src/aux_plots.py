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
    
    
def plot_3scores(s1, s2, s3, lab1, lab2, lab3):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(s1, color='cyan', label=lab1, alpha=0.8)
    ax.plot(s2, color='magenta', label=lab2, alpha=0.5)
    ax.plot(s3, color='red', label=lab3)
    plt.legend()
    
    plt.xlabel("Episode")
    plt.ylabel("Final Score")
    
def plot_3rewards(r1, r2, r3, lab1, lab2, lab3):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
 
    ax.plot(r1, color='cyan', label=lab1, alpha=0.8)
    ax.plot(r2, color='magenta', label=lab2, alpha=0.5)
    ax.plot(r3, color='red', label=lab3)
    plt.legend()
    
    plt.xlabel("Episode")
    plt.ylabel("Final Reward")
