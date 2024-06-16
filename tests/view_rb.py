import pickle
import sys
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/dataset')
from ReplayBuffer import ReplayBufferDataset

file_path = '/home/dyb/Thesis/Isaacmultiarmstandalonedata/experimentsSACIL0508/replaybuffers/Replay_buffer_1.pkl'

with open(file_path, 'rb') as f:
    replay_buffer = pickle.load(f)

for value in replay_buffer:
    print(value)