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
        {'env': 'hopper',
         'mode': 'bn_no_relu',
         'max_epochs': int(1e4),
         'data_size': int(1e5),
         'arch': '256-BR-256-BR-256|T',
         'lamH': 1e-4,
         'lamW': 1e-4,
         'lr': 1e-2,
         'eval_freq': 10,
         'seed': 2,
         'num_eval_batch': 400,
         'group': 'bn_case2'},

        {'env': 'hopper',
         'mode': 'no_relu',
         'max_epochs': int(2e5),
         'data_size': int(1e4),
         'arch': '256-R-256-R-256|T',
         'lamH': -1,
         'lamW': 5e-5,
         'lr': 1e-2,
         'eval_freq': 100,
         'seed': 1,
         'num_eval_batch': 100,
         'group': 'no_bn'}
    ]

    actual_setting = settings[args.setting]

    """replace values & global setup"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_new'
    hyper2logname1 = {
        'env': 'E',
        'max_epochs': 'Eps',
        'data_size': 'DS',
        'lamW': 'W',
        'seed': 's',
    }
    hyper2logname2 = {
        'env': 'E',
        'mode': 'M',
        'max_epochs': 'Eps',
        'data_size': 'DS',
        'lamW': 'wd',
        'seed': 's',
    }

    hyper2logname = hyper2logname1 if config.lamH != -1 else hyper2logname2
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()