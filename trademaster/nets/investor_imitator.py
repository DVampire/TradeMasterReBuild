import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import NETS
from .custom import Net


@NETS.register_module()
class MLPReg(Net):
    def __init__(self, n_input, n_hidden, n_output=1):
        super(MLPReg, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.act = torch.nn.LeakyReLU()
        self.linear1 = nn.Linear(self.n_input, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = nn.Linear(self.n_hidden, n_output)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze()

@NETS.register_module()
class MLPCls(Net):
    def __init__(self, n_input, n_hidden, n_output=1):
        super(MLPCls, self).__init__()
        self.affline1 = nn.Linear(n_input, n_hidden)
        self.affline2 = nn.Linear(n_hidden, n_output)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x = torch.nn.Sigmoid()(x)
        x = self.affline1(x)
        x = torch.nn.Sigmoid()(x)
        action_scores = self.affline2(x).unsqueeze(0)

        return F.softmax(action_scores, dim=1)
