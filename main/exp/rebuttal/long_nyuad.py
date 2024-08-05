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
        'mode', 'M', ['null'],

        'max_epochs', 'Eps', [int(3e6)],
        'batch_size', '', [256],
        'data_size', 'DS', [1000],
        'arch', '', ['256-R-256-R-256-R|T'],
        'normalize', '', ['none'],

        'optimizer', '', ['sgd'],
        'lamH', '', [-1],
        'lamW', 'wd', [0],
        'lr', '', [1e-2],

        'eval_freq', '', [100],
        'seed', 's', [0, 1, 2]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.num_eval_batch = 100
    config.saved_model = f'E{config.env}_Mnull_Eps3000000_DS1000_wd0_s{config.seed}'

    config.data_folder = './dataset/mujoco'
    config.project = 'NC_rebuttal'
    config.group = 'extend'
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items() if v != ''])
    config.name = config.name.replace('3000000', '6000000')

    run_BC(config)


if __name__ == '__main__':
    main()
