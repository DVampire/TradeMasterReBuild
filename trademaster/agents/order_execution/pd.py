import os
import random
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr, get_optim_param
import numpy as np
import torch
from types import MethodType
from copy import deepcopy


class PPOtrainer:
    # PPO1 td error+KL td error is calculated using the new net times a factor calculated by both of the policy

    def __init__(self, net, old_net, optimizer):
        self.net = net
        self.old_net = old_net
        self.old_net.load_state_dict(self.net.state_dict())
        self.optimizer = optimizer

    def choose_action(self, s_public, s_private):
        mu, sigma, V = self.old_net(s_public, s_private)
        dis = torch.distributions.normal.Normal(mu, sigma)
        a = dis.sample()
        log_p = dis.log_prob(a)
        return a.item()

    def get_dis(self, s_public, s_private):
        mu, sigma, V = self.old_net(s_public, s_private)
        dis = torch.distributions.normal.Normal(mu, sigma)
        return dis

    def get_probablity_ratio(self, s_public, s_private, a):
        mu_old, sigma_old, _ = self.old_net(s_public, s_private)
        mu, sigma, _ = self.net(s_public, s_private)
        # print(mu_old.shape)
        # print(sigma_old.shape)
        new_dis = torch.distributions.normal.Normal(mu, sigma)
        old_dis = torch.distributions.normal.Normal(mu_old, sigma_old)
        new_prob = new_dis.log_prob(a).exp()
        # print(new_prob.shape)
        old_prob = old_dis.log_prob(a).exp()
        # print(old_prob.shape)
        return new_prob / (old_prob + 1e-12)

    def get_KL(self, s_public, s_private, a):
        mu_old, sigma_old, _ = self.old_net(s_public, s_private)
        mu, sigma, _ = self.net(s_public, s_private)
        new_dis = torch.distributions.normal.Normal(mu, sigma)
        old_dis = torch.distributions.normal.Normal(mu_old, sigma_old)
        kl = torch.distributions.kl.kl_divergence(new_dis, old_dis)
        return kl

    def choose_action_test(self, s_public, s_private):
        with torch.no_grad():
            mu, sigma, V = self.old_net(s_public, s_private)
        return mu.cpu().squeeze().detach().numpy()

    def uniform(self):
        self.old_net.load_state_dict(self.net.state_dict())


@AGENTS.register_module()
class OrderExecutionPD(AgentBase):
    def __init__(self, **kwargs):
        super(OrderExecutionPD, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)

        '''network'''
        self.act = get_attr(kwargs, "act", None).to(self.device)
        self.cri = get_attr(kwargs, "cri", None) if get_attr(kwargs, "cri", None) else self.act
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        '''optimizer'''
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None) if get_attr(kwargs, "cri_optimizer",
                                                                                 None) else self.act_optimizer
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.student_ppo = PPOtrainer(net=self.act, old_net=self.act_target, optimizer=self.act_optimizer)
        self.teacher_ppo = PPOtrainer(net=self.cri, old_net=self.cri_target, optimizer=self.cri_optimizer)

        self.criterion = get_attr(kwargs, "criterion", None)

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)

        self.memory_student = []
        self.memory_teacher = []

        self.gamma = get_attr(kwargs, "gamma", 0.9)
        self.climp = get_attr(kwargs, "climp", 0.2)
        self.beta = get_attr(kwargs, "beta", 1)
        self.lambada = get_attr(kwargs, "lambada", 1)
        self.save_freq = get_attr(kwargs, "update_freq", 1000)
        self.memory_capacity = get_attr(kwargs, "memory_capacity", 100)
        self.memory_update_freq = get_attr(kwargs, "memory_update_freq", 10)
        self.sample_effiency = get_attr(kwargs, "sample_effiency", 0.5)
        self.memory_size = 0
        self.step_teacher = 0
        self.step_student = 0

    def store_transcation_teacher(self, info, a, r, info_, done):
        self.memory_teacher.append((
            torch.from_numpy(info["perfect_state"]).to(self.device).float(),
            torch.from_numpy(info["private_state"]).to(self.device).float(),
            torch.tensor([a]).to(self.device).float(),
            torch.tensor([r]).to(self.device).float(),
            torch.from_numpy(info_["perfect_state"]).to(self.device).float(),
            torch.from_numpy(info_["private_state"]).to(self.device).float(),
            torch.tensor([done]).to(self.device).float(),
        ))

    def teacher_learn(self):
        perfect_state_list = []
        private_state_list = []
        a_list = []
        r_list = []
        perfect_n_state_list = []
        private_n_state_list = []
        done_list = []
        for perfect_state, private_state, a, r, perfect_n_state, private_n_state, done in self.memory_teacher:
            advangetage = (
                    r + ((self.gamma * self.teacher_ppo.net.get_V(
                perfect_n_state, private_n_state)).squeeze() *
                         (1 - done).squeeze()) - (self.teacher_ppo.net.get_V(
                perfect_state, private_state)).squeeze()).squeeze()
            log_ratio = self.teacher_ppo.get_probablity_ratio(
                perfect_n_state, private_n_state, a)
            # print(log_ratio)
            kl = self.teacher_ppo.get_KL(perfect_n_state, private_n_state, a)
            loss = -(advangetage * log_ratio - self.beta * kl)
            self.teacher_ppo.optimizer.zero_grad()
            loss.backward()
            self.teacher_ppo.optimizer.step()
        self.teacher_ppo.uniform()
        if self.step_teacher % self.memory_update_freq == 1:
            self.memory_teacher = []

        # print(log_ratio)
        #     perfect_state_list.append(perfect_state)
        #     private_state_list.append(private_state)
        #     a_list.append(a)
        #     r_list.append(r)
        #     perfect_n_state_list.append(perfect_n_state)
        #     private_n_state_list.append(private_n_state)
        #     done_list.append(done)
        # perfect_state = torch.cat(perfect_state_list, dim=0)
        # private_state = torch.cat(private_state_list, dim=0)
        # a = torch.cat(a_list, dim=0)
        # r = torch.cat(r_list, dim=0)
        # perfect_n_state = torch.cat(perfect_n_state_list, dim=0)
        # private_n_state = torch.cat(private_n_state_list, dim=0)
        # done = torch.cat(done_list, dim=0)

        # print((self.gamma *
        #        self.teacher_ppo.net.get_V(perfect_n_state, private_n_state) *
        #        (1 - done)).squeeze().shape)
        # print((self.gamma * self.teacher_ppo.net.get_V(
        #     perfect_n_state, private_n_state)).squeeze().shape)
        # print(((self.gamma * self.teacher_ppo.net.get_V(
        #     perfect_n_state, private_n_state)).squeeze() *
        #        (1 - done).squeeze()).shape)

        # advangetage = r + ((self.gamma * self.teacher_ppo.net.get_V(
        #     perfect_n_state, private_n_state)).squeeze() *
        #                    (1 - done).squeeze()) - (self.teacher_ppo.net.get_V(
        #                        perfect_state, private_state)).squeeze()
        # log_ratio = self.teacher_ppo.get_probablity_ratio(
        #     perfect_n_state, private_n_state, a)
        # print(log_ratio.shape)

    def student_learn(self):
        perfect_state_list = []
        private_state_list = []
        a_list = []
        r_list = []
        perfect_n_state_list = []
        private_n_state_list = []
        done_list = []
        for imperfect_state, private_state, perfect_state, a, r, imperfect_n_state, private_n_state, perfect_n_state, done in self.memory_student:
            advangetage = (
                    r + ((self.gamma * self.student_ppo.net.get_V(
                imperfect_n_state, private_n_state)).squeeze() *
                         (1 - done).squeeze()) - (self.student_ppo.net.get_V(
                imperfect_state, private_state)).squeeze()).squeeze()
            log_ratio = self.student_ppo.get_probablity_ratio(
                imperfect_n_state, private_n_state, a)
            # print(log_ratio)
            kl = self.student_ppo.get_KL(imperfect_n_state, private_n_state, a)
            teacher_dis = self.teacher_ppo.get_dis(perfect_state,
                                                   private_n_state)
            student_dis = self.student_ppo.get_dis(imperfect_n_state,
                                                   private_n_state)
            loss = -(
                    advangetage * log_ratio - self.beta * kl - self.lambada *
                    torch.distributions.kl.kl_divergence(teacher_dis, student_dis))
            self.student_ppo.optimizer.zero_grad()
            loss.backward()
            self.student_ppo.optimizer.step()
        self.student_ppo.uniform()
        if self.step_student % self.memory_update_freq == 1:
            self.memory_student = []

    def store_transcation_student(self, s, info, a, r, s_, info_, done):
        self.memory_student.append(
            (torch.from_numpy(s).to(self.device).float(),
             torch.from_numpy(info["private_state"]).to(self.device).float(),
             torch.from_numpy(info["perfect_state"]).to(self.device).float(),
             torch.tensor([a]).to(self.device).float(),
             torch.tensor([r]).to(self.device).float(),
             torch.from_numpy(s_).to(self.device).float(),
             torch.from_numpy(info_["private_state"]).to(self.device).float(),
             torch.from_numpy(info_["perfect_state"]).to(self.device).float(),
             torch.tensor([done]).to(self.device).float()))

    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
