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
         'data_size': 5000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(5e5),
         'lamW': 1.5e-3},
        {'env': 'reacher',
         'data_size': 5000,
         'arch': '256-R-256-R-256-R-256-R-256-R|T',
         'mode': 'big',
         'max_epochs': int(5e5),
         'lamW': 1.5e-3},
        {'env': 'reacher',
         'data_size': 5000,
         'arch': '256-BR-256-BR-256-BR|T',
         'mode': 'bn',
         'max_epochs': int(5e5),
         'lamW': 1.5e-3},

        {'env': 'swimmer',
         'data_size': 100000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(2e4),
         'lamW': 1e-2},
        {'env': 'swimmer',
         'data_size': 100000,
         'arch': '256-R-256-R-256-R-256-R-256-R|T',
         'mode': 'big',
         'max_epochs': int(2e4),
         'lamW': 1e-2},
        {'env': 'swimmer',
         'data_size': 100000,
         'arch': '256-BR-256-BR-256-BR|T',
         'mode': 'bn',
         'max_epochs': int(2e4),
         'lamW': 1e-2},

        {'env': 'hopper',
         'data_size': 100000,
         'arch': '256-R-256-R-256-R|T',
         'max_epochs': int(2e4),
         'lamW': 9.5e-4},
        {'env': 'hopper',
         'data_size': 100000,
         'arch': '256-R-256-R-256-R-256-R-256-R|T',
         'mode': 'big',
         'max_epochs': int(2e4),
         'lamW': 9.5e-4},
        {'env': 'hopper',
         'data_size': 100000,
         'arch': '256-BR-256-BR-256-BR|T',
         'mode': 'bn',
         'max_epochs': int(2e4),
         'lamW': 9.5e-4},
    ]

    actual_setting = settings[args.setting]

    """replace values & global setup"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.lamH = -1
    config.lr = 1e-2

    config.eval_freq = 10

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_explore'
    config.group = 'more_data'
    hyper2logname = {
        'env': 'E',
        'mode': 'M',
        'data_size': 'DS',
        'lamW': 'wd',
        'max_epochs': 'Eps'
    }
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()