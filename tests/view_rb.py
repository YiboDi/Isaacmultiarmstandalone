import pickle
import sys
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/dataset')
from ReplayBuffer import ReplayBufferDataset

file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalonedata/experimentsSACIL0508/replaybuffers/Replay_buffer_1.pkl'

with open(file_path, 'rb') as f:
    replay_buffer = pickle.load(f)

for value in replay_buffer:
    print(value)