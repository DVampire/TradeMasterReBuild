import torch
import torch.nn as nn

from .builder import NETS
from .custom import Net

@NETS.register_module()
class QNet(Net):
    def __init__(self, n_state, n_action, hidden_nodes):
        super().__init__()
        self.fc1 = nn.Linear(n_state, hidden_nodes)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_nodes, n_action)
        self.out.weight.data.normal_(0, 0.1)
        self.out_a = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        actions_value = self.out(x)
        a_task = self.out_a(x)
        return actions_value, a_task
