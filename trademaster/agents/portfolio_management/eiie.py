import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr
import torch
import random


@AGENTS.register_module()
class PortfolioManagementEIIE(AgentBase):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIE, self).__init__()

        self.device = get_attr(kwargs, "device", None)

        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None).to(self.device)

        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)

        self.criterion = get_attr(kwargs, "criterion", None)

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)

        self.memory_counter = 0  # for storing memory
        self.memory_capacity = get_attr(kwargs, "memory_capacity", 1000)
        self.gamma = get_attr(kwargs, "gamma", 0.9)

        self.test_action_memory = []  # to store the
        self.memory_counter = 0
        self.memory_capacity = 1000
        self.s_memory = []
        self.a_memory = []
        self.r_memory = []
        self.sn_memory = []
        self.policy_update_frequency = 500
        self.critic_learn_time = 0

    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri
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

    def store_transition(
            self,
            s,
            a,
            r,
            s_,
    ):  # 定义记忆存储函数 (这里输入为一个transition)
        self.memory_counter = self.memory_counter + 1
        if self.memory_counter < self.memory_capacity:
            self.s_memory.append(s)
            self.a_memory.append(a)
            self.r_memory.append(r)
            self.sn_memory.append(s_)
        else:
            number = self.memory_counter % self.memory_capacity
            self.s_memory[number - 1] = s
            self.a_memory[number - 1] = a
            self.r_memory[number - 1] = r
            self.sn_memory[number - 1] = s_

    def compute_single_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.act(state)
        action = action.detach().cpu().numpy()
        return action

    def learn(self):
        length = len(self.s_memory)
        out1 = random.sample(range(length), int(length / 10))
        # random sample
        s_learn = []
        a_learn = []
        r_learn = []
        sn_learn = []
        for number in out1:
            s_learn.append(self.s_memory[number])
            a_learn.append(self.a_memory[number])
            r_learn.append(self.r_memory[number])
            sn_learn.append(self.sn_memory[number])
        self.critic_learn_time = self.critic_learn_time + 1

        for bs, ba, br, bs_ in zip(s_learn, a_learn, r_learn, sn_learn):
            # update actor
            bs, ba, br, bs_ = bs.to(self.device), ba.to(self.device), br.to(self.device), bs_.to(self.device)
            a = self.act(bs)
            q = self.cri(bs, a)
            a_loss = -torch.mean(q)
            self.act_optimizer.zero_grad()
            a_loss.backward(retain_graph=True)
            self.act_optimizer.step()
            # update critic
            a_ = self.act(bs_)
            q_ = self.cri(bs_, a_.detach())
            q_target = br + self.gamma * q_
            q_eval = self.cri(bs, ba.detach())
            # print(q_eval)
            # print(q_target)
            td_error = self.criterion(q_target.detach(), q_eval)
            # print(td_error)
            self.act_optimizer.zero_grad()
            td_error.backward()
            self.cri_optimizer.step()
