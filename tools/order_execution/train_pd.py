import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import os
import torch

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "order_execution", "order_execution_BTC_pd_pd_adam_mse.py"),
                        help="download datasets config file path")
    parser.add_argument("--task_name", type=str, default="train")
    args = parser.parse_args()
    return args


def test_pd():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    task_name = args.task_name

    cfg = replace_cfg_vals(cfg)
    print(cfg)

    dataset = build_dataset(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))
    valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
    test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))

    n_action = train_environment.action_space.shape[0]
    n_state = train_environment.observation_space.shape[0]
    input_feature = train_environment.observation_space.shape[-1]
    _, info = train_environment.reset()
    private_feature = info["private_state"].shape[-1]

    cfg.net.update(dict(input_feature=input_feature, private_feature=private_feature))

    t_net = build_net(cfg.net)
    t_old_net = build_net(cfg.net)
    s_net = build_net(cfg.net)
    s_old_net = build_net(cfg.net)


    work_dir = os.path.join(ROOT, cfg.trainer.work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    t_optimizer = build_optimizer(cfg, default_args=dict(params=t_net.parameters()))
    s_optimizer = build_optimizer(cfg, default_args=dict(params=s_net.parameters()))
    loss = build_loss(cfg)

    agent = build_agent(cfg, default_args=dict(n_action=n_action,
                                               n_state=n_state,
                                               t_net=t_net,
                                               t_old_net=t_old_net,
                                               s_net=s_net,
                                               s_old_net=s_old_net,
                                               t_optimizer=t_optimizer,
                                               s_optimizer=s_optimizer,
                                               loss=loss,
                                               device=device))

    trainer = build_trainer(cfg, default_args=dict(train_environment=train_environment,
                                                   valid_environment=valid_environment,
                                                   test_environment=test_environment,
                                                   agent=agent,
                                                   device=device,
                                                   ))
    if task_name.startswith("train"):
        trainer.train_and_valid()
        print("train end")
    elif task_name.startswith("test"):
        trainer.test()
        print("test end")


if __name__ == '__main__':
    test_pd()
    """
    algorithmic_trading
    portfolio_management
    """
