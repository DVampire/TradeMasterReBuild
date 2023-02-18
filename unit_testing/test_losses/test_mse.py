import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.losses.builder import build_loss
from trademaster.losses import MSELoss
from trademaster.losses import HFTLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Download Alpaca Datasets")
    parser.add_argument(
        "--config",
        default=osp.join(
            ROOT,
            "configs",
            "high_frequency_trading",
            "high_frequency_trading_BTC_dqn_dqn_adam_mse.py",
        ),
        help="download datasets config file path",
    )
    args = parser.parse_args()
    return args


def test_mse():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)
    print(cfg)

    loss = build_loss(cfg)
    assert isinstance(loss, HFTLoss)
    return loss


if __name__ == "__main__":
    import torch

    loss = test_mse()
    x = torch.randn(32, 1)
    y = torch.randn(32, 1)
    z = (torch.randn(32, 11) + 1e-8).softmax(dim=-1)
    f = (torch.randn(32, 11) + 1e-8).softmax(dim=-1)
    print(loss(x, y, z, f))
