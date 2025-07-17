import os
import torch
import pyrallis
import argparse

from agent.DT import TrainConfig, train_DT
from agent.utils import get_setting_dt

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    setting = args.setting

    settings = [
        'env', 'E', ['door', 'hammer', 'pen', 'relocate'],
        'dataset', 'D', ['human'],
        'weight_decay', 'WD', [0, 3e-4, 3e-3, 3e-2, 3e-1, 0.5, 0.7, 1, 1.5, 2],
        'update_steps', '', [1_000_000]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.task_id = args.setting

    RTG_mapping = {'door': (2900, 1450),
                   'hammer': (12800, 6400),
                   'pen': (3100, 1550),
                   'relocate': (4300, 2150)}
    config.target_returns = RTG_mapping[config.env]

    config.project = 'nrc4rl'
    config.group = 'v1'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    train_DT(config)


if __name__ == '__main__':
    main()
