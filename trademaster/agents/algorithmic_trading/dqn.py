import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr
import numpy as np
import torch

@AGENTS.register_module()
class AlgorithmicTradingDQN(AgentBase):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingDQN, self).__init__()

        self.device = get_attr(kwargs, "device", None)

        self.act_net = get_attr(kwargs, "act_net", None).to(self.device)
        self.cri_net = get_attr(kwargs, "cri_net", None).to(self.device)
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)
        self.loss = get_attr(kwargs, "loss", None)

        self.n_action = get_attr(kwargs, "n_action", None)
        self.n_state = get_attr(kwargs, "n_state", None)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity = get_attr(kwargs, "memory_capacity", 2000)
        self.memory = np.zeros((self.memory_capacity, self.n_state * 2 + 3))
        self.epsilon = get_attr(kwargs,"epsilon", 0.9)
        self.target_freq = get_attr(kwargs,"target_freq",50)
        self.gamma = get_attr(kwargs, "gamma", 0.9)
        self.future_loss_weights = get_attr(kwargs, "future_loss_weights", 0.2)

    def get_save(self):
        models = {
            "act_net":self.act_net,
            "cri_net":self.cri_net
        }
        optimizers = {
            "act_optimizer":self.act_optimizer,
            "cri_optimizer":self.cri_optimizer
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(self.device)
        if np.random.uniform() < self.epsilon:
            actions_value, info = self.act_net.forward(x)
            action = torch.max(actions_value,1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def choose_action_test(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(self.device)
        actions_value, info = self.act_net.forward(x)
        action = torch.max(actions_value,1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def store_transition(self, s, a, r, s_, info):
        transition = np.hstack((s, [a, r, info], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_freq == 0:
            self.cri_net.load_state_dict(
                self.act_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.memory_capacity // 10)

        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.n_state]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.n_state:self.n_state +1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.n_state + 1:self.n_state + 2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_state:]).to(self.device)
        b_info = torch.FloatTensor(b_memory[:,self.n_state + 2:self.n_state + 3]).to(self.device)

        q_eval = self.act_net(b_s)[0].gather(1, b_a)
        v_eval = self.act_net(b_s)[1]
        q_next = self.cri_net(b_s_)[0].detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.memory_capacity // 10, 1)

        loss = self.loss(q_eval, q_target)
        loss_future = self.loss(v_eval, b_info)
        loss = loss + self.future_loss_weights * loss_future

        self.act_optimizer.zero_grad()
        loss.backward()
        self.act_optimizer.step()