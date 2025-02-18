import os
import torch
import pyrallis
import argparse

from main.BC import TrainConfig, run_BC
from main.exp.dt.print_exps import hyper2logname
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
        {'env': 'reacher',
         'max_epochs': int(1.5e6),
         'data_size': 1000,
         'lamW': 1.5e-3,
         'eval_freq': 1.5e4
         },

        {'env': 'swimmer',
         'max_epochs': int(2e5),
         'data_size': 1000,
         'lamW': 1e-2,
         'eval_freq': 2e3
         },

        {'env': 'hopper',
         'max_epochs': int(2e4),
         'data_size': 10000,
         'lamW': 1e-2,
         'eval_freq': 2e2
         }
    ]

    # indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**settings[setting])
    config.device = DEVICE

    config.data_folder = '/NC_regression/dataset/mujoco'
    config.project = 'NC_new'
    config.group = 'data_collection'

    hyper2logname = {'env': 'E',
                     'max_epochs': 'MaxEp',
                     'data_size': 'DS',
                     'lamW': 'WD'}
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items()])

    run_BC(config)


if __name__ == '__main__':
    main()
