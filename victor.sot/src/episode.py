import time

from collections import namedtuple
from typing import List

import src.agents as agents


Step = namedtuple('Step', ['state', 'action', 'reward', 'score'])

class Episode(object):
    """
    An Episode is a representation of a single run of the game.
    It contains all the steps taken: the rewards associated with each
    state-action pairs and the total score at that point.
    You can acess it in a list-like interface: `episode[10].action`
    Note: We use `reward` and `score` here because it allows us to explore 
    different reward strategies.
    """
    def __init__(self):
        self.steps = []
        self.length = 0

    def __iter__(self):
        return self.steps.__iter__()

    def __getitem__(self, i):
        return self.steps[i]

    def add_step(self, state, action, reward, score):
        step = Step(state, action, reward, score)
        self.steps.append(step)
        self.length += 1

    def print_epi(self):
        for s in self.steps:
            print(f"{s.state[:10]}...:\ta {s.action} -> r {int(s.reward)} -> s {int(s.score)}")

    def get_final_score(self):
        return max([s for _, _, _, s in self.steps])

    def get_total_reward(self):
        return sum([r for _, _, r, _ in self.steps])

    def print_final_score(self):
        final_score = self.get_final_score()
        print(f"Final Score at t = {self.length}: {int(final_score)}")


        

def generate_episode(env,
                     reduce_state,
                     reward_policy,
                     agent: agents.Agent,
                     RAM_mask: List[int],
                     render: bool=False) -> Episode:
    """Performs one run of the game and returns an Episode containing all the
    steps taken."""
    epi = Episode()
    game_over = False
    state = env.reset()
    state = reduce_state(state)[RAM_mask].data.tobytes()  # Select useful bytes
    action = agent.act(state)

    score = 0

    while not game_over:
        if render:
            time.sleep(0.005)
            env.render()

        ob, reward, game_over, _ = env.step(action)

        ob = reduce_state(ob)
        #print(len(ob))
        reward = reward_policy(reward, ob, action)
        if reward == reward_policy.REWARD_IF_CROSS:
            score += 1

        epi.add_step(state, action, reward, score)
        state = ob[RAM_mask].data.tobytes()
        action = agent.act(state)  # Next action

    return epi