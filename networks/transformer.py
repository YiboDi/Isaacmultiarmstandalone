
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FlexibleTransformerMLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=None, nhid=None, nlayers=None):
        """
        Initialize the FlexibleTransformerMLPModel.
        :param input_dim: Dimension of the input embeddings.
        :param output_dim: Dimension of the final output vector.
        :param nhead: Number of heads in the multiheadattention models. Default is input_dim // 64.
        :param nhid: Dimension of the feedforward network model in nn.TransformerEncoder. Default is input_dim * 4.
        :param nlayers: Number of nn.TransformerEncoderLayer in nn.TransformerEncoder. Default is 2.
        """
        super(FlexibleTransformerMLPModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Default values if not provided
        if nhead is None:
            nhead = max(1, input_dim // 64)  # Ensure at least 1 head
        if nhid is None:
            nhid = input_dim * 4
        if nlayers is None:
            nlayers = 2

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Adaptive Pooling to handle variable sequence lengths and ensure fixed-size output
        # self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=input_dim)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=256)

        # MLP Layers (assuming the output of adaptive pooling is flattened)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, src):
        """
        Forward pass of the model.
        :param src: Tensor of shape [batch_size, seq_len, input_dim] containing the input embeddings.
        """
        # Adjusting src to shape [seq_len, batch_size, input_dim] for the transformer
        src = src.permute(1, 0, 2)

        # Pass the input through the Transformer encoder
        transformer_output = self.transformer_encoder(src)

        # Permute back to [batch_size, seq_len, input_dim] and apply adaptive pooling
        pooled_output = self.adaptive_pooling(transformer_output.permute(1, 2, 0))

        # Flatten the output of pooling to fit into MLP
        # alternativly use transformer output[0] as the input of mlp
        flattened_output = torch.flatten(pooled_output, start_dim=1)

        # Pass the flattened output through MLP layers to get the final output vector
        output = self.mlp(flattened_output)

        return output
    
class tf_policy(FlexibleTransformerMLPModel):
    def __init__(self, input_dim=107, output_dim=6):
        super().__init__(input_dim, output_dim, nhead, nhid, nlayers)

class tf_q(FlexibleTransformerMLPModel):
    def __init__(self, input_dim=107+6, output_dim=1):
        super().__init__(input_dim, output_dim, nhead, nhid, nlayers)