import sys
import torch

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/networks')
from lstm_torch_layers import WholeNet

from BaseNet import StochasticActor


def test_whole_net():
    # Define a mock observation space (adjust dimensions as needed)
    observation_space = torch.randn(3, 107)  # Example: 1 sequence of length 10

    # Create an instance of WholeNet
    net = WholeNet(observation_space)

    # Create a sample observation tensor
    observation = torch.randn(3, observation_space.shape[-1])  # Match the last dimension of observation_space

    # Forward pass through the network
    output = net(observation)

    # Check the output shape and type
    print("Output shape:", output.shape)
    print("Output type:", type(output))

    # Add more tests here as needed

# def test_BaseNet():


# Run the test function
if __name__ == "__main__":
    test_whole_net()