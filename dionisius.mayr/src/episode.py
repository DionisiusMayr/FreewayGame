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
        step = Step(state.data.tobytes(), action, reward, score)
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


        

def generate_episode(env, agent: agents.Agent, RAM_mask: List[int], render: bool=False) -> Episode:
    """Performs one run of the game and returns an Episode containing all the
    steps taken."""
    epi = Episode()
    game_over = False
    state = env.reset()[RAM_mask]  # Select useful bytes
    action = agent.act(state, 0)  # TODO: aren't reducing the dimensionality
                                  # of the first action, but it shouldn't
                                  # impact the final result
    score = 0

    while not game_over:
        if render:
            time.sleep(0.005)
            env.render()

        ob, reward, game_over, _ = env.step(action)

        # Doesn't matter where we were hit
        ob[16] = 1 if ob[16] != 255 else 0

        # Reduce chicken y-position
        ob[14] = ob[14] // 3

        # The chicken is in the x-posistion ~49
        # We don't need to represent cars far from the chicken
        for i in range(108, 118):
            if ob[i] < 20 or ob[i] > 80:
                ob[i] = 0
            else:
                # Reduce the cars x-positions sample space
                ob[i] = ob[i] // 3

        if reward == 1:
            score += 1
        elif ob[16] == 1:  # Collision!
            reward = -1
#         elif reward != 1 and action != 1:  # Don't incentivate staying still
#             reward -= 0.2

        epi.add_step(state, action, reward, score)
        state = ob[RAM_mask]
        action = agent.act(state, reward)  # Next action

    return epi