import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, load_model, load_best_model, save_model, save_best_model
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
        self.seeds_list = get_attr(kwargs, "seeds_list", [12345])

        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        self.set_seed(random.choice(self.seeds_list))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True

    def train_and_valid(self):
        valid_score_list = []

        for epoch in range(1, self.epochs+1):
            print("Train Episode: [{}/{}]".format(epoch, self.epochs))
            stacked_state = []
            s = self.train_environment.reset()
            stacked_state.append(s)
            for i in range(self.state_length - 1):
                action = np.array([0, 0])
                s, r, done, _ = self.train_environment.step(action)
                stacked_state.append(s)

            action = self.agent.compute_action(stacked_state)

            episode_reward_sum = 0
            count = 0
            while True:
                count = count + 1
                old_states = []
                for state in stacked_state.copy():
                    state = torch.from_numpy(state).reshape(1, -1).float()
                    old_states.append(state)
                old_states = torch.cat(old_states, dim=0).float().to(self.device)
                action = self.agent.compute_action(stacked_state)
                s_new, reward, done, _ = self.train_environment.step(action)

                episode_reward_sum += reward

                stacked_state.pop(0)
                stacked_state.append(s_new)
                new_states = []
                for state in stacked_state.copy():
                    state = torch.from_numpy(state).reshape(1, -1).float()
                    new_states.append(state)
                new_states = torch.cat(new_states, dim=0).float().to(self.device)

                self.agent.save_transication(
                    old_states,
                    torch.from_numpy(action).reshape(-1).float().to(self.device),
                    torch.tensor(reward).float().reshape(-1).to(self.device),
                    new_states,
                    0,
                    torch.tensor(done).float().reshape(-1).to(self.device))

                if count % 100 == 1:
                    self.agent.learn()
                    self.agent.inputs = []
                    self.agent.actions = []
                    self.agent.rewards = []
                    self.agent.next_states = []
                    self.agent.previous_rewards = []
                    self.agent.dones = []

                if done:
                    print("Train Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break

            save_model(self.checkpoints_path,
                       epoch=epoch,
                       save=self.agent.get_save())

            print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
            stacked_state = []
            s = self.valid_environment.reset()
            stacked_state.append(s)
            for i in range(self.state_length - 1):
                action = np.array([0, 0])
                s, r, done, _ = self.valid_environment.step(action)
                stacked_state.append(s)

            episode_reward_sum = 0
            while True:
                action = self.agent.compute_action_test(stacked_state)
                s_new, reward, done, _ = self.valid_environment.step(action)
                stacked_state.pop(0)
                stacked_state.append(s_new)
                episode_reward_sum += reward
                if done:
                    print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break
            valid_score_list.append(episode_reward_sum)

        max_index = np.argmax(valid_score_list)
        save_best_model(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            save=self.agent.get_save()
        )

    def test(self):
        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        stacked_state = []
        s = self.test_environment.reset()
        stacked_state.append(s)
        for i in range(self.state_length - 1):
            action = np.array([0, 0])
            s, r, done, _ = self.test_environment.step(action)
            stacked_state.append(s)

        episode_reward_sum = 0
        while True:
            action = self.agent.compute_action_test(stacked_state)
            s_new, reward, done, _ = self.test_environment.step(action)
            stacked_state.pop(0)
            stacked_state.append(s_new)
            episode_reward_sum += reward
            if done:
                print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        result = np.array(self.test_environment.portfolio_value_history)
        np.save(os.path.join(self.work_dir,"result.npy"), result)
        return episode_reward_sum
