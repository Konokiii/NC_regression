import os
import torch
import argparse

from agent.any_percent_bc import TrainConfig, run_BC
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
        'env', '', ['halfcheetah'],
        'max_epochs', '', [int(6e4)],
        'data_size', '', [0.1],
        'lamW', '', [0, 0.00001, 0.0001, 0.001, 0.01],
        'arch', '', ['3-64-R', '3-128-R', '3-256-R', '3-512-R', '3-1024-R',
                     '1-256-R', '2-256-R', '4-256-R', '5-256-R', '5-1024-R'
                     ],
        'eval_freq', '', [int(3e2)]
    ]

    indexes, actual_setting, total, _ = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    # config = TrainConfig(**settings[setting])
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.batch_size = 4096

    config.data_folder = './dataset'
    config.project_folder = './'
    config.project = 'NC_new'
    config.group = 'data_collection'

    hyper2logname = {'env': 'E',
                     'max_epochs': 'MaxEp',
                     'data_size': 'DS',
                     'lamW': 'WD',
                     'arch': 'A'}
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items()])

    run_BC(config)


if __name__ == '__main__':
    main()

