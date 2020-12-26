# TODO: We might need to implement another method to "train" the agent.
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
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


class MonteCarloAprox(Agent):
    def __init__(self, gamma: float, available_actions: int, N0: float,nFeatures,scaler=StandardScaler(),feat_type=None):

        self.gamma = gamma
        self.available_actions = available_actions
        self.W=np.zeros(nFeatures+2)
        self.scaler=scaler
        self.Returns = defaultdict(lambda: defaultdict(list))
        self.state_visits = defaultdict(lambda: 0)
        self.N0 = N0
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))
        self.feat_type=feat_type
    def act(self, stateByte,stateValue):
        "the epsilon for the first episodio will choose randomly"
        #visits_on_state = sum([len(v) for k, v in self.Returns[state].items()])
        epsilon = self.N0 / (self.N0 + self.state_visits[stateByte])
        if np.random.choice(np.arange(self.available_actions), p=[1 - epsilon, epsilon]):
            action = np.random.choice(self.available_actions)  # Explore!
        elif self.state_visits[stateByte] == 0:
            action = 1  # Bias toward going forward
        else:
            action = np.argmax([self.getApproximation(stateValue, act) for act in range(self.available_actions)])  # Greedy action
        return action
    def createFeature(self,state,action):
        
        if self.feat_type == 'all':
            #Transforms the state from bytes to integers and concatenates with the action
            feat_state = self.scaler.transform(state.reshape(1, -1))
            return np.append(feat_state, [action,1])
        elif self.feat_type == 'mean':
            feat_state = np.concatenate((state[0:2], np.mean(state[2:]), np.count_nonzero(state[2:])), axis=None)
            feat_state = self.scaler.transform(feat_state.reshape(1,-1))
            return  np.append(feat_state, [action,1])
            
    def getApproximation(self, state, action):
        feature = self.createFeature(state, action)
        return np.dot(feature, self.W)

    def trainScaler(self, env, mask, n_samples=10000):
        if self.feat_type == 'all':
            self.scaler.fit(np.array([env.observation_space.sample()[mask] for x in range(n_samples)]))
        elif self.feat_type == 'mean':
            observations = [env.observation_space.sample()[mask] for x in range(n_samples)]
            features = [np.concatenate((state[0:2], np.mean(state[2:]), np.count_nonzero(state[2:])), axis=None)
            for state in observations]
            self.scaler.fit(np.array(features))
    def update_W(self, stateValue, action, reward):
        # fixed value of alpha
        alpha = 0.00001#(1 / self.Nsa[state][action])
        #stateValue form sArray
        self.W = self.W + alpha*(reward  - self.getApproximation(stateValue, action))*self.createFeature(stateValue, action)

    def updating(self, episode):
        G = 0
        # S:State in Bytes
        # S_array: State in Values provisional solutions to reconstrution mistake from buffer
        S = np.array([s for s, _, _, _,_ in episode])
        S_array = np.array([sa for _, sa, _, _,_ in episode])
        A=np.array([a for _,_, a, _, _ in episode])
        R = np.array([r for _, _,_, r, _ in episode])
       
        #self.fit_normalizer(episode=episode,typeFeature=typeFeature,printed=False,scaler=scaler)
        for t in range(episode.length - 1):
            # count of visits to apply the approximate function
            self.state_visits[S[t]] += 1
            self.Nsa[S[t]][A[t]] += 1
            # first visit to the state   
            if self.state_visits[S[t]]==1:
                G=sum(R[t:])
                #print(S_array[t])
                self.update_W(stateValue =S_array[t],action= A[t], reward =G)
                #print('Update in '+str(t)+ " :", self.W)
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


if __name__ == '__main__':
    print('Testing agents.py...')
    agent = Baseline()
    print('All good!')