import torch
import torch.nn as nn
from .builder import NETS
from .custom import Net

@NETS.register_module()
class EIIEConv(Net):
    def __init__(self,
                 n_input,
                 n_output = 1,
                 length = None,
                 kernel_size = 3,
                 num_layer=None,
                 n_hidden=None):
        super(EIIEConv, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.length = length
        self.kernel_size = kernel_size
        self.act = torch.nn.ReLU(inplace=False)
        self.con1d = nn.Conv1d(self.n_input,
                               self.n_output,
                               kernel_size=3)
        self.con2d = nn.Conv1d(self.n_output,
                               1,
                               kernel_size=self.length - self.kernel_size + 1)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.con1d(x)
        x = self.act(x)
        x = self.con2d(x)
        x = self.act(x)
        x = self.con3d(x)
        x = x.view(-1)

        # self.linear2 = nn.Linear(len(x), len(x) + 1)
        # x = self.linear2(x)
        x = torch.cat((x, self.para), dim=0)
        x = torch.softmax(x, dim=0)

        return x

@NETS.register_module()
class EIIELSTM(Net):
    def __init__(self,
                 n_input,
                 n_output = 1,
                 length = None,
                 kernel_size = 3,
                 num_layer=None,
                 n_hidden=None):
        super(EIIELSTM, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=n_input,
                            hidden_size=self.n_hidden,
                            num_layers=self.num_layer,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, self.n_output)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1, :]).view(-1, 1, 1)
        x = self.con3d(x)
        x = x.view(-1)
        x = torch.cat((x, self.para), dim=0)
        x = torch.softmax(x, dim=0)
        return x

@NETS.register_module()
class EIIERNN(Net):
    def __init__(self,
                 n_input,
                 n_output=1,
                 length=None,
                 kernel_size=3,
                 num_layer=None,
                 n_hidden=None
                 ):
        super(EIIERNN, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.num_layer = num_layer
        self.rnn = nn.RNN(input_size=self.n_input,
                          hidden_size=self.n_hidden,
                          num_layers=self.num_layer,
                          batch_first=True)
        self.linear = nn.Linear(self.n_hidden, self.n_output)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x):
        lstm_out, _ = self.rnn(x)
        x = self.linear(lstm_out[:, -1, :]).view(-1, 1, 1)
        x = self.con3d(x)
        x = x.view(-1)
        x = torch.cat((x, self.para), dim=0)
        x = torch.softmax(x, dim=0)
        return x

@NETS.register_module()
class EIIECritic(Net):
    def __init__(self,
                 n_input,
                 n_output=1,
                 length=None,
                 kernel_size=3,
                 num_layer=None,
                 n_hidden=None
                 ):
        super(EIIECritic, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=self.n_input,
                            hidden_size=self.n_hidden,
                            num_layers=self.num_layer,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, self.n_output)
        self.con3d = nn.Conv1d(1, 1, kernel_size=1)
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, x, a):
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1, :]).view(-1, 1, 1)
        x = self.con3d(x)
        x = x.view(-1)
        x = torch.cat((x, self.para, a), dim=0)
        x = torch.nn.ReLU(inplace=False)(x)
        number_nodes = len(x)
        self.linear2 = nn.Linear(number_nodes, 1).to(x.device)
        x = self.linear2(x)
        return x