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
    def act(self, ob, reward):
        pass


class Baseline(Agent):
    """
    The Baseline agent always move up, regardless of the reward received.
    """
    def __init__(self):
        pass
    
    def act(self, ob, reward):
        return 1  # Always move up!


class MonteCarloControl(Agent):
    def __init__(self, epsilon: float, gamma: float, available_actions: int):
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(available_actions))
        # TODO: Are we able to use numpy arrays for `Returns`?
        self.Returns = defaultdict(lambda: defaultdict(list))
        self.pi = defaultdict(lambda: 1)

    def act(self, ob, reward):
        """
        With prob (1 - epsilon) execute the greedy action, and with
        probability epsilon, execute a random action.
        """
        if np.random.choice(np.arange(2), p=[1 - self.epsilon, self.epsilon]):
            return np.random.choice(2) # Going back is not an option
        else:
            return self.pi[ob.data.tobytes()]

    def update_policy(self, episode):
        G = 0
        S = np.array([s for s, _, _, _ in episode])
        A = np.array([a for _, a, _, _ in episode])
        R = np.array([r for _, _, r, _ in episode])

        for t in reversed(range(episode.length - 1)):
            # TODO: add the action to this comment
            # TODO: According to the algorithm I should check if S_t appers in
            #  the sequence S_0, S_1, S_2, ..., S_t-1.
            G = self.gamma * G + R[t + 1]
            self.Returns[S[t]][A[t]].append(G)
            self.Q[S[t]][A[t]] = sum(self.Returns[S[t]][A[t]]) / len(self.Returns[S[t]][A[t]])  # Mean
            self.pi[S[t]] = self.Q[S[t]].argmax()

#         print(f"Pi: {len(pi):8} ", end='')#, Q: {len(Q)}, Returns: {len(Returns)}")

        return episode.get_final_score(), episode.get_total_reward()


if __name__ == '__main__':
    print('Testing agents.py...')
    agent = Baseline()
    print('All good!')