import torch
import torch.nn as nn


class ReliabilityHead(nn.Module):
    """
    Reliability prediction MLP.

    Input:
        [conf, area, aspect_ratio, entropy, modality]

    Output:
        reliability score ∈ [0,1]
    """

    def __init__(self, input_dim=5):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)