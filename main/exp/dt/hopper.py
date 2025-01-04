import os
import torch
import pyrallis
import argparse

from main.DT import TrainConfig, train_DT
from main.utils import get_setting_dt

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
        'env', 'E', ['hopper'],
        'dataset', 'D', ['medium-replay', 'medium', 'medium-expert'],
        'weight_decay', 'WD', [0, 3e-5, 3e-4, 3e-3, 3e-2, 3e-1],
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE

    config.project = 'nrc4rl'
    config.group = 'test'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    train_DT(config)


if __name__ == '__main__':
    main()