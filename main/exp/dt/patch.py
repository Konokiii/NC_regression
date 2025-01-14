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

    # settings = [
    #     'env', 'E', ['hopper'],
    #     'dataset', 'D', ['medium-replay', 'medium', 'medium-expert', 'expert'],
    #     'weight_decay', 'WD', [0, 3e-4, 3e-3, 3e-2, 3e-1, 0.5, 0.7, 1, 1.5, 2],
    #     'update_steps', '', [500_000]
    # ]

    # indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    settings = [{'dataset': 'expert', 'weight_decay': 0},
                {'dataset': 'expert', 'weight_decay': 3e-4},
                {'dataset': 'expert', 'weight_decay': 3e-2},
                {'dataset': 'expert', 'weight_decay': 0.5},
                {'dataset': 'expert', 'weight_decay': 1.5},
                {'dataset': 'medium-expert', 'weight_decay': 3e-4},
                {'dataset': 'medium-expert', 'weight_decay': 0.5},
                {'dataset': 'medium-expert', 'weight_decay': 1},
                {'dataset': 'medium', 'weight_decay': 0},
                {'dataset': 'medium', 'weight_decay': 0.5},
                {'dataset': 'medium', 'weight_decay': 0.7},
                {'dataset': 'medium-replay', 'weight_decay': 0},
                {'dataset': 'medium-replay', 'weight_decay': 0.3},
                ]
    actual_setting = settings[setting]
    hyper2logname = {'env': 'E', 'dataset': 'D', 'weight_decay': 'WD'}

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.task_id = args.setting

    config.env = 'hopper'
    config.update_steps = 2_000_000

    config.project = 'nrc4rl'
    config.group = 'v1'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    train_DT(config)


if __name__ == '__main__':
    main()
