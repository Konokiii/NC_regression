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
    setting = args.setting

    settings = [
        'env', 'E', ['hopper'],
        'mode', 'M', ['no_relu', 'null'],

        'max_epochs', '', [250],
        'batch_size', '', [256],
        'data_size', 'DS', [1000, 5000, 10000],
        'arch', '', ['256-R-256-R-256-R|T'],
        'normalize', '', ['none'],

        'optimizer', '', ['sgd'],
        'lamH', '', [-1],
        'lamW', 'W', [1e-2,
                      9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1.8e-3, 1.5e-3, 1.2e-3, 1e-3,
                      9.5e-4, 9e-4, 8e-4],
        'lr', 'lr', [1e-2],

        'eval_freq', '', [1],
        'seed', '', [0]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.num_eval_batch = 100

    if config.data_size == 1000:
        config.max_epochs = int(1.2e6)
    elif config.data_size == 5000:
        config.max_epochs = int(2.4e5)
    elif config.data_size == 10000:
        config.max_epochs = int(1.2e5)

    config.eval_freq = 100

    if config.mode == 'no_relu':
        config.arch = '256-R-256-R-256|T'
    elif config.mode == 'gelu':
        config.arch = '256-R-256-R-256-G|T'

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_5000_sgd'
    config.group = 'final'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()