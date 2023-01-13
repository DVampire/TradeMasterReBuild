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
class OrderExecutionPDTrainer(Trainer):
    def __init__(self, **kwargs):
        super(OrderExecutionPDTrainer, self).__init__()
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.device = get_attr(kwargs, "device", None)
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

        valid_score_list = []
        valid_number = 0
        for i in range(self.epochs):
            print("Train Episode: [{}/{}]".format(i+1, self.epochs))
            s, info = self.train_environment.reset()
            # train the teacher first
            done = False
            while not done:
                public_state = torch.from_numpy(info["perfect_state"]).to(
                    self.device).float()
                private_state = torch.from_numpy(info["private_state"]).to(
                    self.device).float()
                self.agent.step_teacher += 1

                action = self.agent.teacher_ppo.choose_action(
                    public_state, private_state)
                s, r, done, info_ = self.train_environment.step(action)
                self.agent.store_transcation_teacher(info, action, r, info_, done)
                info = info_
                if self.agent.step_teacher % self.agent.memory_capacity == 1:
                    print("teacher learning")
                    self.agent.teacher_learn()
            #then train the student
            s, info = self.train_environment.reset()
            done = False
            while not done:
                public_state = torch.from_numpy(s).to(self.device).float()
                private_state = torch.from_numpy(info["private_state"]).to(
                    self.device).float()
                self.agent.step_student += 1

                action = self.agent.student_ppo.choose_action(
                    public_state, private_state)
                s_, r, done, info_ = self.train_environment.step(action)
                self.agent.store_transcation_student(s, info, action, r, s_, info_,
                                               done)
                info = info_
                s = s_

                if self.agent.step_student % self.agent.memory_capacity == 1:
                    print("student learning")
                    self.agent.student_learn()

                if self.agent.step_student % self.agent.save_freq == 1:
                    torch.save(
                        self.agent.student_ppo.old_net, os.path.join(self.all_model_path,"{}_net.pth".format(valid_number)))
                    valid_number += 1
                    print("Valid Episode: [{}/{}]".format(i + 1, self.epochs))
                    s, info = self.valid_environment.reset()
                    done = False
                    while not done:
                        public_state = torch.from_numpy(s).to(
                            self.device).float()

                        private_state = torch.from_numpy(
                            info["private_state"]).to(self.device).float()
                        action = self.agent.student_ppo.choose_action_test(
                            public_state, private_state)

                        s_, r, done, info_ = self.valid_environment.step(action)
                        info = info_
                        s = s_
                    valid_score_list.append(self.valid_environment.money_sold)
                    break
        index = valid_score_list.index(max(valid_score_list))
        net_path = os.path.join(self.all_model_path, "{}_net.pth".format(index))
        self.agent.student_ppo.old_net = torch.load(net_path)
        torch.save(self.agent.student_ppo.old_net, os.path.join(self.best_model_path,"best_net.pth"))

    def test(self):
        self.agent.student_ppo.old_net = torch.load(os.path.join(self.best_model_path,"best_net.pth"))

        s, info = self.test_environment.reset()
        action_list = []
        reward_list = []

        done = False
        while not done:
            public_state = torch.from_numpy(s).to(self.device).float()
            private_state = torch.from_numpy(info["private_state"]).to(
                self.device).float()
            action = self.agent.student_ppo.choose_action_test(
                public_state, private_state)
            s_, r, done, info_ = self.test_environment.step(action)
            info = info_
            s = s_
            action_list.append(action)
            reward_list.append(r)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        np.save(os.path.join(self.work_dir,"action.npy"), action_list)
        np.save(os.path.join(self.work_dir,"reward.npy"), reward_list)
