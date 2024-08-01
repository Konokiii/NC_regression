import os
import torch
import pyrallis
import argparse

from main.BC import TrainConfig, run_BC
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

    settings = [
        {'env': 'reacher',
         'data_size': 1000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(1.2e6),
         'lamW': 0,
         'seed': 1},

        {'env': 'reacher',
         'data_size': 1000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(2e6),
         'lamW': 0,
         'seed': 0},

        {'env': 'swimmer',
         'data_size': 1000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(2e6),
         'lamW': 0,
         'seed': 0},

        {'env': 'hopper',
         'data_size': 10000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(4e5),
         'lamW': 0,
         'seed': 0}
    ]

    actual_setting = settings[args.setting]

    """replace values & global setup"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.lamH = -1
    config.lr = 1e-2

    config.eval_freq = 100

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_rebuttal'
    config.group = 'first'
    hyper2logname = {
        'env': 'E',
        'mode': 'M',
        'max_epochs': 'Eps',
        'data_size': 'DS',
        'lamW': 'wd',
        'seed': 's',
    }
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()