import torch
import torch.nn as nn


class LSTMExtractor(nn.Module):
    def __init__(self, observation_space, hidden_size=256, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.seq_len = observation_space.shape[0]
        input_size = observation_space.shape[-1]

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, observation, batch_size = 1):
        # LSTM expects input of shape (batch, seq_len, features)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(observation.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(observation.device)
        input = observation.unsqueeze(0)

        # Forward propagate LSTM
        features, _ = self.lstm(input, (h0, c0))

        return features


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # layers = []
        # for in_size, out_size in zip(sizes[:-1], sizes[1:]):
        #     layers.append(nn.Linear(in_size, out_size))
        #     layers.append(nn.ReLU())
        # self.mlp = nn.Sequential(*layers)
        # self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.mlp(x)
    
class WholeNet(nn.Module):
    def __init__(self, observation_space, hidden_size=256, num_layers=1) -> None:
        # observation space should change accorrding to the task
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.seq_len = observation_space.shape[0]
        input_size = observation_space.shape[-1]

        self.mlp = ActorCritic()
        self.lstm = LSTMExtractor(observation_space=observation_space)

    def forward(self, observation):
        #observation shape should also change with the task
        #seq_len * input_size = num_ur5 * 107

        # Forward propagate LSTM
        lstm_out = self.lstm(observation)

        # Reshape output from hidden cell into desired feature size
        output = self.mlp(lstm_out[:, -1, :])

        return output