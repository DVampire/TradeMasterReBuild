from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr
import numpy as np
import os
import pandas as pd

@TRAINERS.register_module()
class OrderExecutionETEOTrainer(Trainer):
    def __init__(self, **kwargs):
        super(OrderExecutionETEOTrainer, self).__init__()
        self.device = get_attr(kwargs, "device", None)
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.state_length = self.train_environment.state_length
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.all_model_path = os.path.join(self.work_dir, "all_model")
        self.best_model_path = os.path.join(self.work_dir, "best_model")
        if not os.path.exists(self.all_model_path):
            os.makedirs(self.all_model_path)
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

    def train_and_valid(self):
        reward_list = []

        for i in range(self.epochs):
            print("Train Episode: [{}/{}]".format(i+1, self.epochs))
            num_epoch = i
            stacked_state = []
            s = self.train_environment.reset()
            stacked_state.append(s)
            for i in range(self.state_length - 1):
                action = np.array([0, 0])
                s, r, done, _ = self.train_environment.step(action)
                stacked_state.append(s)
            action = self.agent.compute_action(stacked_state)
            done = False
            i = 0
            while not done:
                i = i + 1
                old_states = []
                for state in stacked_state.copy():
                    state = torch.from_numpy(state).reshape(1, -1).float()
                    old_states.append(state)
                old_states = torch.cat(old_states, dim=0).float().to(self.device)
                action = self.agent.compute_action(stacked_state)
                s_new, reward, done, _ = self.train_environment.step(action)
                stacked_state.pop(0)
                stacked_state.append(s_new)
                new_states = []
                for state in stacked_state.copy():
                    state = torch.from_numpy(state).reshape(1, -1).float()
                    new_states.append(state)
                new_states = torch.cat(new_states, dim=0).float().to(self.device)
                self.agent.save_transication(
                    old_states,
                    torch.from_numpy(action).reshape(-1).float().to(
                        self.device),
                    torch.tensor(reward).float().reshape(-1).to(self.device),
                    new_states, 0,
                    torch.tensor(done).float().reshape(-1).to(self.device))
                if i % 100 == 1:
                    print("updating")
                    self.agent.learn()
                    self.agent.inputs = []
                    self.agent.actions = []
                    self.agent.rewards = []
                    self.agent.next_states = []
                    self.agent.previous_rewards = []
                    self.agent.dones = []

            torch.save(self.agent.act_net, os.path.join(self.all_model_path, "policy_state_value_net_{}.pth".format(num_epoch)))
            stacked_state = []
            print("Valid Episode: [{}/{}]".format(i + 1, self.epochs))
            s = self.valid_environment.reset()
            stacked_state.append(s)
            for i in range(self.state_length - 1):
                action = np.array([0, 0])
                s, r, done, _ = self.valid_environment.step(action)
                stacked_state.append(s)
            done = False
            while not done:
                action = self.agent.compute_action_test(stacked_state)
                s_new, reward, done, _ = self.valid_environment.step(action)
                stacked_state.pop(0)
                stacked_state.append(s_new)
            reward_list.append(reward)
        max_reward = max(reward_list)
        index = reward_list.index(max_reward)

        net_path = os.path.join(self.all_model_path, "policy_state_value_net_{}.pth".format(index))
        self.agent.cri_net = torch.load(net_path)
        torch.save(self.agent.cri_net, os.path.join(self.best_model_path,"policy_state_value_net.pth"))

    def test(self):
        self.agent.cri_net = torch.load(os.path.join(self.best_model_path,"policy_state_value_net.pth"))
        stacked_state = []
        s = self.test_environment.reset()
        stacked_state.append(s)
        for i in range(self.state_length - 1):
            action = np.array([0, 0])
            s, r, done, _ = self.test_environment.step(action)
            stacked_state.append(s)
        done = False
        while not done:
            action = self.agent.compute_action_test(stacked_state)
            s_new, reward, done, _ = self.test_environment.step(action)
            stacked_state.pop(0)
            stacked_state.append(s_new)
            if done:
                final_reward=reward
        result = np.array(self.test_environment.portfolio_value_history)
        np.save(os.path.join(self.work_dir,"result.npy"), result)
        return final_reward
