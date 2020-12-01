# TODO: We might need to implement another method to "train" the agent.
from abc import ABC
from abc import abstractmethod

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
    def act(self, ob, reward, game_over):
        pass

class Baseline(Agent):
    """The Baseline agent always move up, regardless of the reward received."""
    def __init__(self):
        pass
    
    def act(self, ob, reward, game_over):
        return 1  # Always move up!

    
if __name__ == '__main__':
    print('Testing agents.py...')
    agent = Baseline()
    print('All good!')