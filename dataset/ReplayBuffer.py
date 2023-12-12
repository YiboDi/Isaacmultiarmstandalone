from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor



class ReplayBufferDataset(Dataset):
    def __init__(self, data, device, capacity):
        self.keys = list(data)
        self.data = data
        self.device = device
        self.capacity = int(capacity)
        self.observations_padded = None
        self.next_observations_padded = None
        self.freshness = 0.0

    def __len__(self):
        return len(self.data['observations'])

    def extend(self, data):
        len_new_data = len(data['observations'])
        for key in self.data:
            self.data[key].extend(data[key])
        if len(self) != 0:
            self.freshness += len_new_data / len(self)
        else:
            self.freshness = 0.0

        if len(self.data['observations']) > self.capacity:
            amount = len(self.data['observations']) - self.capacity
            for key in self.data:
                del self.data[key][:amount]

        self.padded_observations = None
        self.next_observations_padded = None

    def pad_observations(self, input):
        return pad_sequence(input, batch_first=True)

    def __getitem__(self, idx):
        if self.observations_padded is None or\
                self.next_observations_padded is None:
            self.observations_padded = self.pad_observations(
                self.data['observations'])
            self.next_observations_padded = self.pad_observations(
                self.data['next_observations'])
            if 'critic_observations' in self.data\
                    and len(self.data['critic_observations']) > 0:
                self.critic_observations_padded = self.pad_observations(
                    self.data['critic_observations'])
                self.critic_next_observations_padded = self.pad_observations(
                    self.data['critic_next_observations'])
            else:
                self.critic_observations_padded = \
                    self.observations_padded
                self.critic_next_observations_padded =\
                    self.next_observations_padded
        if 'critic_observations' in self.data:
            return (
                self.critic_observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.data['actions'][idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['rewards'][idx])).to(
                    self.device, non_blocking=True),
                self.critic_next_observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.next_observations_padded[idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['is_terminal'][idx])).to(
                    self.device, non_blocking=True)
            )
        else:
            return (
                self.observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.data['actions'][idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['rewards'][idx])).to(
                    self.device, non_blocking=True),
                self.next_observations_padded[idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['is_terminal'][idx])).to(
                    self.device, non_blocking=True)
            )

