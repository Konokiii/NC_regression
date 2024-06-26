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
        'env', 'E', ['swimmer', 'reacher'],

        'max_epochs', 'T', [200],
        'batch_size', 'B', [256],
        'arch', 'A', ['256-256'],

        'reg_coff_H', 'coffH', [-1, 0, 1e-2, 1e-3, 1e-4, 1e-5],
        'reg_coff_W', 'coffW', [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        'lr', 'lr', [3e-4],

        'eval_freq', 'Hz', [1],
        'seed', 'S', [0]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.num_eval_batch = 100
    config.data_ratio = 0.1 if config.env == 'swimmer' else 1.0

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.group = 'official'
    config.name = '_'.join([v + str(actual_setting[k]) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
