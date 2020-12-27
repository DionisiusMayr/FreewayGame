# Need to `pip install dill` first!
import dill
import pickle
from datetime import datetime


PATH = './serialized_models/'

class Experiment():
    def __init__(self, agent, scores, total_rewards, reduce_state, reward_policy):
        self.agent = agent
        self.scores = scores
        self.total_rewards = total_rewards
        self.reduce_state = reduce_state
        self.reward_policy = reward_policy

    def print_parameters(self):
        self.agent.print_parameters()

        print(f"reward_policy.REWARD_IF_CROSS = {self.reward_policy.REWARD_IF_CROSS}")
        print(f"reward_policy.REWARD_IF_COLISION = {self.reward_policy.REWARD_IF_COLISION}")
        print(f"reward_policy.REWARD_IF_STILL = {self.reward_policy.REWARD_IF_STILL}")

    def _generate_filename(self, algo: str) -> str:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return f"{PATH}{algo}_{now}.dill"

    def save_experiment(self, algo: str):
        fn = self._generate_filename(algo)
        with open(fn, "wb") as f:
            dill.dump(obj=self, file=f)

    @classmethod
    def load_experiment(self, fn: str):
        with open(fn, "rb") as f:
            return dill.load(file=f)