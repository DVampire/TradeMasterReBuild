import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net

@NETS.register_module()
class ETEOStacked(Net):
    def __init__(self, length, features, action_dim = 2, nodes=128):
        # nodes is a list where the element reprensents the nodes on each layer
        super(ETEOStacked, self).__init__()
        self.length = length
        self.nodes = nodes
        self.linear1 = nn.Linear(features * self.length, 128)
        self.linear2 = nn.Linear(128, 128)
        self.act_linear_volume = nn.Linear(128, 2)
        self.act_linear_price = nn.Linear(128, 2)
        self.v_linear = nn.Linear(128, 1)
        self.act = torch.relu

    def forward(self, x):
        x = x.reshape(1, -1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        action_volume = self.act_linear_volume(x)
        action_price = self.act_linear_price(x)
        v = self.v_linear(x)

        return action_volume, action_price, v

@NETS.register_module()
class ETEOLSTM(Net):
    def __init__(self, length, features, action_dim=2, nodes=128):
        super(ETEOLSTM, self).__init__()
        self.length = length
        self.nodes = nodes
        self.action_dim = action_dim
        self.linear = nn.Linear(features, nodes)
        self.linear2 = nn.Linear(nodes, nodes)
        self.act = torch.relu
        self.lstm = nn.LSTM(input_size=nodes,
                            hidden_size=nodes,
                            num_layers=1,
                            batch_first=True)
        self.linear_volume = nn.Linear(nodes + self.action_dim, 2)
        self.linear_price = nn.Linear(nodes + self.action_dim, 2)
        self.linear_value = nn.Linear(nodes + self.action_dim, 1)

    def forward(self, x: torch.tensor, previous_action: torch.tensor):
        # here is a little difference from the origional article, it put both action and reward on the last time_stamp into the output of
        #lstm but since the setting here is that the reward is sparase, there is no need to add previous reward to it because it is always zero
        #(the termial state is the only one with none-zero reward but it is the last state, it could be any state's previous state)
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        previous_action = previous_action.reshape(1, -1)
        x = torch.cat((x, previous_action), dim=1)
        action_volume = self.linear_volume(x)
        action_price = self.linear_price(x)
        v = self.linear_value(x)
        return action_volume, action_price, v