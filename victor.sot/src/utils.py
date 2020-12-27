import pickle5 as pickle

def loadHistory(fileDir):
    with open(fileDir, 'rb') as handle:
        data = pickle.load(handle)
        return data
def get_score_rewards(score,rewards):
  s= loadHistory(score)
  r= loadHistory(rewards)
  return s,r