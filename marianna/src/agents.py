# TODO: We might need to implement another method to "train" the agent.
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

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

        self.Q = defaultdict(lambda: np.zeros(self.available_actions))
        # TODO: Are we able to use numpy arrays for `Returns`?
        self.Returns = defaultdict(lambda: defaultdict(list))
        self.pi = defaultdict(lambda: 1)  # Forward Bias
        self.N0 = N0

    def act(self, state):
        visits_on_state = sum([len(v) for k, v in self.Returns[state].items()])
        epsilon = self.N0 / (self.N0 + visits_on_state)
 
        if np.random.choice(np.arange(self.available_actions), p=[1 - epsilon, epsilon]):
            return np.random.choice(self.available_actions)  # Explore!
        else:
            return self.pi[state]  # Greedy

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
            # Alpha is the `len(self.Returns[S[t]][A[t]])`
            self.Q[S[t]][A[t]] = sum(self.Returns[S[t]][A[t]]) / len(self.Returns[S[t]][A[t]])  # Mean
            self.pi[S[t]] = self.Q[S[t]].argmax()

#         print(f"Pi: {len(pi):8} ", end='')#, Q: {len(Q)}, Returns: {len(Returns)}")

        return episode.get_final_score(), episode.get_total_reward()


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
#         elif self.state_visits[state] == 0:
        elif self.Q[state].max() == 0.0 and self.Q[state].min() == 0.0:
            action = 1  # Bias toward going forward
        else:
            action = self.Q[state].argmax()  # Greedy action

        self.state_visits[state] += 1

        self.Nsa[state][action] += 1

        return action

    def update_Q(self, old_state, new_state, action, reward):
#         if reward:
#             print("Old Q[state][action]", self.Q[old_state][action])
#             print(f"alpha {self.alpha}, reward {reward}, gamma {self.gamma}, right side {(reward + (self.gamma * self.Q[new_state].max()) - self.Q[old_state][action])}")
        alpha = (1 / self.Nsa[old_state][action])
        self.Q[old_state][action] = self.Q[old_state][action] + alpha * (reward + (self.gamma * self.Q[new_state].max()) - self.Q[old_state][action])
#         if reward:
#             print("New Q[state][action]", self.Q[old_state][action])

class QLearningLinearApprox(Agent):
    def __init__(self, gamma: float, available_actions: int, N0: float, weights_length: int):
        self.gamma = gamma
        self.available_actions = available_actions
        self.N0 = N0

        # self.Q = defaultdict(lambda: np.zeros(self.available_actions))
        # np.random.seed(42)
        self.W = np.zeros(weights_length+2)#np.random.normal(0,1, weights_length)
        self.state_visits = defaultdict(lambda: 0)
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))
        self.scaler = StandardScaler(with_mean=False)
    
    def trainScaler(self, env, mask, n_samples=10000):
        self.scaler.fit(np.array([env.observation_space.sample()[mask] for x in range(n_samples)]))

    def act(self, state):
        epsilon = self.N0 / (self.N0 + self.state_visits[state])

        if np.random.choice(np.arange(self.available_actions), p=[1 - epsilon, epsilon]):
            action = np.random.choice(self.available_actions)  # Explore!
#         elif self.state_visits[state] == 0:
        elif self.state_visits[state] == 0:
            action = 1  # Bias toward going forward
        else:
            #We are assuming that self.available_actions has, at least, two actions
            action = np.argmax([self.getApproximation(state, act) for act in range(self.available_actions)])  # Greedy action

        self.state_visits[state] += 1

        self.Nsa[state][action] += 1

        return action

    def createFeature(self, state, action):
        #Transforms the state from bytes to integers and concatenates with the action
        feat_state = self.scaler.transform(np.frombuffer(state, dtype=np.uint8, count=-1).reshape(1,-1))
        return np.append(feat_state, [action,1])

    def getApproximation(self, state, action):
        feature = self.createFeature(state, action)
        return np.dot(feature, self.W)

    def update_W(self, old_state, new_state, action, reward):
#         if reward:
#             print("Old Q[state][action]", self.Q[old_state][action])
#             print(f"alpha {self.alpha}, reward {reward}, gamma {self.gamma}, right side {(reward + (self.gamma * self.Q[new_state].max()) - self.Q[old_state][action])}")
        alpha = (1 / self.Nsa[old_state][action])
        max_value =  np.max([self.getApproximation(new_state, act) for act in range(self.available_actions)])

        # print("new:{}, {} | old: {}, {}".format(self.createFeature(new_state, action),self.getApproximation(new_state, action), self.createFeature(old_state, action), self.getApproximation(old_state, action)))
        print('Alpha:{}, Action:{}, Reward:{} | Max_value:{} | Update: {}'.format(alpha,action, reward, max_value, alpha*(reward + (self.gamma * max_value) - self.getApproximation(old_state, action))))
        print("Weight:", self.W)
        self.W = self.W + alpha*(reward + (self.gamma * max_value) - self.getApproximation(old_state, action))*self.createFeature(old_state, action)
        # self.Q[old_state][action] = self.Q[old_state][action] + alpha * (reward + (self.gamma * self.Q[new_state].max()) - self.Q[old_state][action])




if __name__ == '__main__':
    print('Testing agents.py...')
    agent = Baseline()
    print('All good!')