from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        layers: List[int] = [64, 128, 64],
    ):
        """ Create instance of model
        :param state_size: (int): size of state
        :param action_size: (int) size of action
        :param seed: (int) random seed
        :param hidden_layer_sizes: List[int]
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.fc4 = nn.Linear(layers[2], action_size)

    def forward(self, state):
        """ Network that calculates actions from states
        :param state: Iterable[float] represents state
        :return: action: Iterable[int] represents possible actions
        """
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
