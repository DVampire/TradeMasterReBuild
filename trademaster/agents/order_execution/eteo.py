import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr
import numpy as np
import torch
from random import sample


@AGENTS.register_module()
class OrderExecutionETEO(AgentBase):
    def __init__(self, **kwargs):
        super(OrderExecutionETEO, self).__init__()

        self.device = get_attr(kwargs, "device", None)

        self.act_net = get_attr(kwargs, "act_net", None).to(self.device)
        self.cri_net = get_attr(kwargs, "cri_net", None).to(self.device)

        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)
        self.loss = get_attr(kwargs, "loss", None)

        self.n_action = get_attr(kwargs, "n_action", None)
        self.n_state = get_attr(kwargs, "n_state", None)

        self.gamma = get_attr(kwargs, "gamma", 0.9)
        self.climp = get_attr(kwargs, "climp", 0.2)
        self.memory_capacity = get_attr(kwargs, "memory_capacity", 1000)
        self.sample_effiency = get_attr(kwargs, "sample_effiency", 0.5)
        self.memory_size = 0

        self.inputs = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.previous_rewards = []
        self.stacked_state = []
        self.dones = []

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

    def compute_action(self, stacked_state):
        # stacked_state is a list of the previous state,(np.array with shape (156,)), whose length is 10
        list_states = []
        for state in stacked_state:
            state = torch.from_numpy(state).reshape(1, -1).float()
            list_states.append(state)
        list_states = torch.cat(list_states, dim=0).to(self.device)
        action_volume, action_price, v = self.act_net(list_states)
        action_volume = action_volume.squeeze()
        action_price = action_price.squeeze()
        v = v.squeeze(0)
        dis_volume = torch.distributions.normal.Normal(
            torch.relu(action_volume[0]) + 0.001,
            torch.relu(action_volume[1]) + 0.001)
        dis_price = torch.distributions.normal.Normal(
            torch.relu(action_price[0]) + 0.001,
            torch.relu(action_price[1]) + 0.001)
        volume = dis_volume.sample()
        price = dis_price.sample()
        action = np.array([torch.abs(volume).item(), torch.abs(price).item()])
        return action

    def compute_action_test(self, stacked_state):
        # stacked_state is a list of the previous state,(np.array with shape (156,)), whose length is 10
        list_states = []
        for state in stacked_state:
            state = torch.from_numpy(state).reshape(1, -1).float()
            list_states.append(state)
        list_states = torch.cat(list_states, dim=0).to(self.device)
        action_volume, action_price, v = self.act_net(list_states)
        action_volume = action_volume.squeeze()
        action_price = action_price.squeeze()
        v = v.squeeze(0)
        action = np.array([
            torch.relu(action_volume[0]).item() + 0.001,
            torch.relu(action_price[0]).item() + 0.001
        ])
        return action

    def save_transication(self, s, a, r, s_, r_previous, done):
        # here, the s,a,r,s_,r_previous are all torch tensor and in the GPU
        # self.memory_size = self.memory_size + 1
        if self.memory_size <= self.memory_capacity:
            self.inputs.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s_)
            self.previous_rewards.append(r_previous)
            self.dones.append(done)
        else:
            index = self.memory_size % self.memory_capacity
            self.inputs[index - 1] = s
            self.actions[index - 1] = a
            self.rewards[index - 1] = r
            self.next_states[index - 1] = s_
            self.previous_rewards[index - 1] = r_previous
            self.dones[index - 1] = done

    def learn(self):
        inputs = []
        actions = []
        rewards = []
        next_states = []
        previous_rewards = []
        dones = []
        number_sample = int(len(self.inputs) * self.sample_effiency)
        sample_list_number = sample(range(len(self.inputs)), number_sample)

        for i in sample_list_number:
            inputs.append(self.inputs[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            next_states.append(self.next_states[i])
            previous_rewards.append(self.previous_rewards[i])
            dones.append(self.dones[i])

        for input, action, reward, next_state, previous_reward, done in zip(inputs, actions, rewards, next_states, previous_rewards,dones):

            action_volume, action_price, v = self.act_net(next_state)

            td_target = reward + self.gamma * v * (1 - done)
            action_volume, action_price, v = self.act_net(input)
            action_volume, action_price, v = action_volume.squeeze(
            ), action_price.squeeze(), v.squeeze(0)
            mean = torch.cat(
                (action_volume[0].unsqueeze(0), action_price[0].unsqueeze(0)))
            std = torch.cat((torch.relu(action_volume[1].unsqueeze(0)) + 0.001,
                             torch.relu(action_price[1].unsqueeze(0)) + 0.001))
            old_dis = torch.distributions.normal.Normal(mean, std)
            log_prob_old = old_dis.log_prob(action).float()
            log_prob_old = (log_prob_old[0] + log_prob_old[1]).float()
            action_volume, action_price, v_s = self.cri_net(next_state)
            action_volume, action_price, v = self.cri_net(input)
            # td_error = torch.min(reward + self.gamma * v_s * (1 - done) - v,
            #                      torch.tensor([100]))
            td_error = reward + self.gamma * v_s * (1 - done) - v
            td_error = td_error.reshape(-1)

            # here is a little different from the original PPO, because there is a processure of passing the td error to different
            # state, however, we are only use 1 state at one time and do the update, therefore, we are simpling use the td error
            # we use td error instead of A to do the optimization
            action_volume, action_price, v = self.cri_net(input)
            action_volume, action_price, v = action_volume.squeeze(
            ), action_price.squeeze(), v.squeeze(0)
            mean = torch.cat(
                (action_volume[0].unsqueeze(0), action_price[0].unsqueeze(0)))
            std = torch.cat((torch.relu(action_volume[1].unsqueeze(0)) + 0.001,
                             torch.relu(action_price[1].unsqueeze(0)) + 0.001))

            new_dis = torch.distributions.normal.Normal(mean, std)
            log_prob_new = new_dis.log_prob(action).float()
            log_prob_new = log_prob_new[0].float() + log_prob_new[1].float()

            ratio = torch.exp(torch.min(log_prob_new - log_prob_old, torch.tensor([10]).to(self.device)))
            L1 = ratio * td_error.float()
            L2 = torch.clamp(ratio, 1 - self.climp,
                             1 + self.climp) * td_error.float()
            loss_pi = -torch.min(L1, L2).mean().float()
            # loss_pi = torch.min(loss_pi, torch.tensor([100000000]))
            loss_v = torch.min(
                torch.nn.functional.mse_loss(td_target.detach().reshape(-1), v.reshape(-1).float()), torch.tensor([1000000000]).to(self.device))
            loss_v = torch.nn.functional.mse_loss(
                td_target.detach().reshape(-1),
                v.reshape(-1).float())
            loss = loss_pi.float() + loss_v.float()

            self.act_optimizer.zero_grad()
            loss.backward()
            self.act_optimizer.step()

        self.act_net.load_state_dict(self.cri_net.state_dict(), strict=True)
