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
        'env', 'E', ['reacher', 'swimmer', 'hopper'],
        'mode', '', ['bn_no_relu'],

        'max_epochs', 'Eps', [int(1e4)],
        'batch_size', '', [256],
        'data_size', 'DS', [int(1e5)],
        'arch', '', ['256-BR-256-BR-256|T'],
        'normalize', '', ['none'],

        'optimizer', '', ['sgd'],
        'lamH', 'H', [-1],
        'lamW', 'W', [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 0],
        'lr', '', [1e-2],

        'eval_freq', '', [10],
        'seed', 's', [0]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.num_eval_batch = 400

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_new'
    config.group = 'bn_case1_explore'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])

    run_BC(config)


if __name__ == '__main__':
    main()
