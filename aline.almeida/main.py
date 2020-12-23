"""
 Useful script to run the experiment from the terminal.
"""

import sys
sys.path.append('../')  # Enable importing from `src` folder

import gym
import time
import src.agents as agents
import src.environment as environment
import src.utils as utils

def main():
    """Change the parameters here to perform your experiment!"""
    environment.run(agents.Baseline, render=False, n_runs=5)

if __name__ == '__main__':
    main()
