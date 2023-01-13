from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr
import numpy as np
import os
import pandas as pd


def make_market_information(df, technical_indicator):
    # based on the information, calculate the average for technical_indicator to present the market average
    all_dataframe_list = []
    index_list = df.index.unique().tolist()
    index_list.sort()
    for i in index_list:
        information = df[df.index == i]
        new_dataframe = []
        for tech in technical_indicator:
            tech_value = np.mean(information[tech])
            new_dataframe.append(tech_value)
        all_dataframe_list.append(new_dataframe)
    new_df = pd.DataFrame(all_dataframe_list,
                          columns=technical_indicator).values
    # new_df.to_csv(store_path)
    return new_df


def make_correlation_information(df: pd.DataFrame, feature="adjclose"):
    # based on the information, we are making the correlation matrix(which is N*N matric where N is the number of tickers) based on the specific
    # feature here,  as default is adjclose
    df.sort_values(by='tic', ascending=True, inplace=True)
    array_symbols = df['tic'].values

    # get data, put into dictionary then dataframe
    dict_sym_ac = {}  # key=symbol, value=array of adj close
    for sym in array_symbols:
        dftemp = df[df['tic'] == sym]
        dict_sym_ac[sym] = dftemp['adjcp'].values

    # create correlation coeff df
    dfdata = pd.DataFrame.from_dict(dict_sym_ac)
    dfcc = dfdata.corr().round(2)
    dfcc = dfcc.values
    return dfcc


@TRAINERS.register_module()
class PortfolioManagementDeepTraderTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementDeepTraderTrainer, self).__init__()

        self.kwargs = kwargs
        self.device = get_attr(kwargs, "device", None)

        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.all_model_path = os.path.join(self.work_dir, "all_model")
        best_model_path = os.path.join(self.work_dir, "best_model")
        if not os.path.exists(self.all_model_path):
            os.makedirs(self.all_model_path)
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

    def train_and_valid(self):

        rewards_list = []
        for i in range(self.epochs):
            print("Train Episode: [{}/{}]".format(i+1, self.epochs))
            j = 0
            done = False
            s = self.train_environment.reset()
            while not done:
                old_asset_state = s
                old_market_state = torch.from_numpy(
                    make_market_information(
                        self.train_environment.data,
                        technical_indicator=self.train_environment.tech_indicator_list)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_old = make_correlation_information(
                    self.train_environment.data)
                weights = self.agent.compute_weights_train(
                    s,
                    make_market_information(
                        self.train_environment.data,
                        technical_indicator=self.train_environment.tech_indicator_list),
                    corr_matrix_old)
                action_asset = self.agent.act_net(
                    torch.from_numpy(old_asset_state).float().to(self.device),
                    corr_matrix_old)
                action_market = self.agent.market_net(old_market_state)
                s, reward, done, _ = self.train_environment.step(weights)
                new_asset_state = s
                new_market_state = torch.from_numpy(
                    make_market_information(
                        self.train_environment.data,
                        technical_indicator=self.train_environment.tech_indicator_list)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_new = make_correlation_information(
                    self.train_environment.data)
                self.agent.store_transition(
                    torch.from_numpy(old_asset_state).float().to(self.device),
                    action_asset,
                    torch.tensor(reward).float().to(self.device),
                    torch.from_numpy(new_asset_state).float().to(self.device),
                    old_market_state, action_market, new_market_state,
                    corr_matrix_old, corr_matrix_new, self.agent.roh_bar)
                j = j + 1
                if j % 100 == 10:
                    self.agent.learn()
            torch.save(
                self.agent.act_net,
                os.path.join(self.all_model_path, "act_net_num_epoch_{}.pth".format(i)))
            torch.save(
                self.agent.cri_net,
                os.path.join(self.all_model_path, "cri_net_num_epoch_{}.pth".format(i)))
            torch.save(
                self.agent.market_net,
                os.path.join(self.all_model_path, "market_policy_num_epoch_{}.pth".format(i)))
            print("Valid Episode: [{}/{}]".format(i + 1, self.epochs))
            s = self.valid_environment.reset()
            done = False
            rewards = 0
            while not done:
                old_state = s
                old_market_state = torch.from_numpy(
                    make_market_information(
                        self.valid_environment.data,
                        technical_indicator=self.valid_environment.tech_indicator_list)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_old = make_correlation_information(
                    self.valid_environment.data)
                weights = self.agent.compute_weights_test(
                    s,
                    make_market_information(
                        self.valid_environment.data,
                        technical_indicator=self.valid_environment.tech_indicator_list),
                    corr_matrix_old)
                s, reward, done, _ = self.valid_environment.step(weights)
                rewards += reward
            rewards_list.append(rewards)
        index = rewards_list.index(np.max(rewards_list))
        act_net_model_path = os.path.join(self.all_model_path, "act_net_num_epoch_{}.pth".format(
            index))
        cri_net_model_path = os.path.join(self.all_model_path, "cri_net_num_epoch_{}.pth".format(
            index))
        market_net_model_path = os.path.join(self.all_model_path, "market_net_num_epoch_{}.pth".format(
            index))

        self.agent.act_net = torch.load(act_net_model_path)
        self.agent.cri_net = torch.load(cri_net_model_path)
        self.agent.market_net = torch.load(market_net_model_path)

        torch.save(self.agent.act_net, os.path.join(self.best_model_path, "act_net.pth"))
        torch.save(self.agent.cri_net, os.path.join(self.best_model_path, "cri_net.pth"))
        torch.save(self.agent.market_net, os.path.join(self.best_model_path, "market_net.pth"))

    def test(self):
        self.agent.act_net = torch.load(os.path.join(self.best_model_path, "act_net.pth"))
        self.agent.cri_net = torch.load(os.path.join(self.best_model_path, "cri_net.pth"))
        self.agent.market_net = torch.load(os.path.join(self.best_model_path, "market_net.pth"))

        s = self.test_environment.reset()
        done = False
        while not done:
            corr_matrix_old = make_correlation_information(
                self.test_environment.data)
            weights = self.agent.compute_weights_test(
                s,
                make_market_information(
                    self.test_environment.data,
                    technical_indicator=self.test_environment.tech_indicator_list),
                corr_matrix_old)
            s, reward, done, _ = self.test_environment.step(weights)
        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
