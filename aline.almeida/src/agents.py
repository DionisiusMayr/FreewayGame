# TODO: We might need to implement another method to "train" the agent.
from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import numpy as np

class Agent(ABC):
    """
    Abstract class to implement agents.
    It requires an `__init__` method to set the required parameters (such
     as epsilon) and an `act` method that implements the policy of the agent.
    """
    @abstractmethod
    def __init__(self, **params):
        pass
    
    @abstractmethod
    def act(self, state):
        pass


class Baseline(Agent):
    """
    The Baseline agent always move up, regardless of the reward received.
    """
    def __init__(self):
        pass
    
    def act(self, state):
        return 1  # Always move up!


class MonteCarloControl(Agent):
    def __init__(self, gamma: float, available_actions: int, N0: float):
        self.gamma = gamma
        self.available_actions = available_actions
        self.N0 = N0

        self.Q = defaultdict(lambda: np.zeros(self.available_actions))
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))
        self.state_visits = defaultdict(lambda: 0)
        self.pi = defaultdict(lambda: 1)  # Forward Bias

    def act(self, state):
        epsilon = self.N0 / (self.N0 + self.state_visits[state])
 
        if np.random.choice(np.arange(self.available_actions), p=[1 - epsilon, epsilon]):
            action = np.random.choice(self.available_actions)  # Explore!
        else:
            action = self.pi[state]  # Greedy

        return action

    def update_policy(self, episode):
        G = 0
        S = np.array(episode.S)
        A = np.array(episode.A)
        R = np.array(episode.R)

        for t in reversed(range(episode.length - 1)):
            self.state_visits[S[t]] += 1
            self.Nsa[S[t]][A[t]] += 1

            alpha = (1 / self.Nsa[S[t]][A[t]])
            G = self.gamma * G + R[t + 1]

            self.Q[S[t]][A[t]] += alpha * (G - self.Q[S[t]][A[t]])
            self.pi[S[t]] = self.Q[S[t]].argmax()

#         print(f"Pi: {len(pi):8} ", end='')

        return episode.get_final_score(), episode.get_total_reward()

    def print_parameters(self):
        print(f"gamma = {self.gamma}")
        print(f"available_actions = {self.available_actions}")
        print(f"N0 = {self.N0}")

class QLearning(Agent):
    def __init__(self, gamma: float, available_actions: int, N0: float):
        self.gamma = gamma
        self.available_actions = available_actions
        self.N0 = N0

        self.Q = defaultdict(lambda: np.zeros(self.available_actions))
        self.state_visits = defaultdict(lambda: 0)
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))

    def act(self, state):
        epsilon = self.N0 / (self.N0 + self.state_visits[state])

        if np.random.choice(np.arange(self.available_actions), p=[1 - epsilon, epsilon]):
            action = np.random.choice(self.available_actions)  # Explore!
        elif self.Q[state].max() == 0.0 and self.Q[state].min() == 0.0:
            action = 1  # Bias toward going forward
        else:
            action = self.Q[state].argmax()  # Greedy action

        self.state_visits[state] += 1
        self.Nsa[state][action] += 1

        return action

    def update_Q(self, old_state, new_state, action, reward):
        alpha = (1 / self.Nsa[old_state][action])
        self.Q[old_state][action] += alpha * (reward + (self.gamma * self.Q[new_state].max()) - self.Q[old_state][action])

    def print_parameters(self):
        print(f"gamma = {self.gamma}")
        print(f"available_actions = {self.available_actions}")
        print(f"N0 = {self.N0}")

class SarsaLambda(Agent):
    def __init__(self, gamma: float, available_actions: int, N0: float, lambd: float):
        self.gamma = gamma
        self.available_actions = available_actions
        self.N0 = N0
        self.lambd = lambd

        self.Q = defaultdict(lambda: np.zeros(self.available_actions))
        self.state_visits = defaultdict(lambda: 0)
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))
        self.E = defaultdict(lambda: np.zeros(self.available_actions))

    def act(self, state):
        epsilon = self.N0 / (self.N0 + self.state_visits[state])

        if np.random.choice(np.arange(self.available_actions), p=[1 - epsilon, epsilon]):
            action = np.random.choice(self.available_actions)  # Explore!
        elif self.Q[state].max() == 0.0 and self.Q[state].min() == 0.0:
            action = 1  # Bias toward going forward
        else:
            action = self.Q[state].argmax()  # Greedy action

        self.state_visits[state] += 1
        self.Nsa[state][action] += 1
        self.E[state][action] += 1

        return action

    def update_Q(self, old_s, new_s, old_a, new_a, reward):
        delta = reward + self.gamma * self.Q[new_s][new_a] - self.Q[old_s][old_a]
        alpha = (1 / self.Nsa[old_s][old_a])

        for s in self.E:
            for a in range(self.available_actions):
                self.Q[old_s][old_a] += alpha * delta * self.E[s][a]
                self.E[s][a] = self.gamma * self.lambd * self.E[s][a]

    def reset_E(self):
        self.E = defaultdict(lambda: np.zeros(self.available_actions))

    def print_parameters(self):
        print(f"gamma = {self.gamma}")
        print(f"available_actions = {self.available_actions}")
        print(f"N0 = {self.N0}")
        print(f"lambd = {self.lambd}")